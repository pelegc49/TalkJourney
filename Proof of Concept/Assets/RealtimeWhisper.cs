using UnityEngine;
using UnityEngine.UI;
using Unity.InferenceEngine;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Unity.Collections;
using Newtonsoft.Json;
using System.Linq;
using System.Text.RegularExpressions; 
using Unity.Profiling;

#if ENABLE_INPUT_SYSTEM 
using UnityEngine.InputSystem;
#endif

public class RealtimeWhisper : MonoBehaviour
{
    [Header("Model Assets")]
    public ModelAsset audioDecoder1;
    public ModelAsset audioDecoder2;
    public ModelAsset audioEncoder;
    public ModelAsset logMelSpectro;
    public TextAsset vocabAsset;

    [Header("Microphone Settings")]
    public int recordingDuration = 30; 
    public KeyCode triggerKey = KeyCode.Space;

    [Header("VAD Settings")]
    [Range(0.0f, 1.0f)]
    public float volumeThreshold = 0.2f;   
    public float requiredSilence = 1f;    

    [Header("Languages & Context")]
    public bool useHebrew = false; 
    public string[] englishSentences; 
    public string[] hebrewSentences; 
    
    [Tooltip("0.0 = Exact match required. 0.4 = Allows loose matching.")]
    [Range(0.0f, 1.0f)]
    public float matchTolerance = 0.3f; 
    
    [Header("Generation Settings")]
    public int maxTokens = 30;

    [Header("Output")]
    public string fullText = ""; 
    public bool isThinking = false;

    // Internal State
    private float silenceTimer = 0f;
    private bool isSpeaking = false;
    private int vadStartPos = 0;
    private int vadEndPos = 0;
    
    // Context
    private string[] _currentSentences;
    private string[] _cleanedSentences;  // Cache cleaned sentences to avoid redundant cleaning
    private readonly StringBuilder _sb = new StringBuilder();
    private readonly StringBuilder _transcriptBuilder = new StringBuilder(256);
    private readonly StringBuilder _sanitizeBuilder = new StringBuilder(256);
    private int _tokensLength;  // Cache tokens.Length to avoid repeated field access

    // Workers
    private Worker encoderEngine, decoderEngine1, decoderEngine2, spectroEngine, argmaxEngine;
    
    // Data
    private NativeArray<int> outputTokens; 
    private NativeArray<int> singleTokenArray; 
    private readonly List<Tensor> _kvTensorsToDispose = new List<Tensor>(16);
    
    private AudioClip micClip;
    private string micDevice;
    private const int SampleRate = 16000; 
    private const int VADSampleWindow = 128;
    private const int MaxAudioSeconds = 30;
    private bool isTranscribing = false;
    private float[] _vadWaveData;
    private float[] _speechDataBuffer;
    private float[] _paddedAudioBuffer;
    private float[] _wrapReadBuffer;
    private int[] _levenshteinPrev;
    private int[] _levenshteinCurr;
    private int _lastPaddedLength;
    private readonly bool[] _allowedAscii = new bool[128];
    private readonly HashSet<int> _recentTokens = new HashSet<int>(16);

    // Threading removed - Sentis GPU operations require main thread
    // Strategy: Use yield returns in TranscribeRoutine to keep cube smooth

    private const string BasicSymbols = "!@#$%^&*()-_=+[]{};:'\"\\|,.<>/?`~";

    private static readonly ProfilerMarker UpdateMarker = new ProfilerMarker("RealtimeWhisper.Update");
    private static readonly ProfilerMarker VadMarker = new ProfilerMarker("RealtimeWhisper.VAD");
    private static readonly ProfilerMarker TranscribeMarker = new ProfilerMarker("RealtimeWhisper.Transcribe");
    private static readonly ProfilerMarker DecodeMarker = new ProfilerMarker("RealtimeWhisper.Decode");

    // Constants
    const int END_OF_TEXT = 50257;
    const int START_OF_TRANSCRIPT = 50258;
    const int ENGLISH = 50259;
    const int HEBREW = 50279; 
    const int TRANSCRIBE = 50359;
    const int NO_TIME_STAMPS = 50363;

    private string[] tokens;
    private int[] whiteSpaceCharacters = new int[256];
    private bool[] _basicSymbolsLookup = new bool[256];  // Fast O(1) symbol lookup for ASCII range

    void Start()
    {
        SetupWhiteSpaceShifts();
        SetupBasicSymbolsLookup();  // Initialize symbol lookup table
        LoadVocab();
        SetupEngines();
        SetupMicrophone();

        _vadWaveData = new float[VADSampleWindow];
        _paddedAudioBuffer = new float[MaxAudioSeconds * SampleRate];
        
        int tokenBufferLength = Mathf.Max(100, maxTokens + 4);
        outputTokens = new NativeArray<int>(tokenBufferLength, Allocator.Persistent);
        singleTokenArray = new NativeArray<int>(1, Allocator.Persistent);
        singleTokenArray[0] = NO_TIME_STAMPS;

        SetLanguage(useHebrew);
    }

    public void SetLanguage(bool isHebrew)
    {
        useHebrew = isHebrew;
        _currentSentences = isHebrew ? hebrewSentences : englishSentences;
        
        // Pre-clean all sentences to avoid redundant cleaning in FindBestMatch
        if (_currentSentences != null)
        {
            _cleanedSentences = new string[_currentSentences.Length];
            for (int i = 0; i < _currentSentences.Length; i++)
            {
                _cleanedSentences[i] = CleanString(_currentSentences[i]);
            }
        }
        
        Debug.Log($"Language switched to: {(useHebrew ? "Hebrew" : "English")}");
    }

    private void OnValidate()
    {
        if (Application.isPlaying)
        {
            SetLanguage(useHebrew);
        }
    }

    void SetupMicrophone()
    {
        if (Microphone.devices.Length > 0)
        {
            micDevice = Microphone.devices[0];
            micClip = Microphone.Start(micDevice, true, recordingDuration, SampleRate);
            Debug.Log($"Microphone Initialized: {micDevice} @ {SampleRate}Hz");
        }
        else
        {
            Debug.LogError("No Microphone detected!");
        }
    }

    void Update()
    {
        using (UpdateMarker.Auto())
        {
        if (isTranscribing) return;
        if (micClip == null || string.IsNullOrEmpty(micDevice)) return;

        bool manualTrigger = false;
        #if ENABLE_INPUT_SYSTEM 
            if (Keyboard.current != null && Keyboard.current.spaceKey.wasPressedThisFrame) manualTrigger = true;
        #else
            if (Input.GetKeyDown(triggerKey)) manualTrigger = true;
        #endif

        if (manualTrigger)
        {
            int now = Microphone.GetPosition(micDevice);
            int start = now - (recordingDuration * SampleRate); 
            if (start < 0) start += micClip.samples;
            StartCoroutine(TranscribeRoutine(start, now));
            return;
        }

        float currentVol = GetCurrentVolume();

        if (currentVol > volumeThreshold)
        {
            if (!isSpeaking) 
            {
                isSpeaking = true;
                int frequency = micClip.frequency;
                int offset = (int)(0.4f * frequency); 
                vadStartPos = Microphone.GetPosition(micDevice) - offset;
                if (vadStartPos < 0) vadStartPos += micClip.samples;
            }
            silenceTimer = 0f; 
        }
        else if (isSpeaking)
        {
            silenceTimer += Time.deltaTime;
            if (silenceTimer > requiredSilence)
            {
                isSpeaking = false;
                vadEndPos = Microphone.GetPosition(micDevice);
                StartCoroutine(TranscribeRoutine(vadStartPos, vadEndPos));
            }
        }
        }
    }

    float GetCurrentVolume()
    {
        using (VadMarker.Auto())
        {
            int micPos = Microphone.GetPosition(micDevice);
            if (micPos < VADSampleWindow + 1) return 0f;
            micClip.GetData(_vadWaveData, micPos - VADSampleWindow);
            float max = 0f;
            for (int i = 0; i < VADSampleWindow; i++)
            {
                float absValue = Mathf.Abs(_vadWaveData[i]);
                if (absValue > max) max = absValue;
            }
            return max;
        }
    }

    IEnumerator TranscribeRoutine(int startSample, int endSample)
    {
        using (TranscribeMarker.Auto())
        {
        isTranscribing = true;
        isThinking = true;
        fullText = "";

        int totalSamples = micClip.samples;
        if (startSample < 0) startSample = 0;
        if (startSample >= totalSamples) startSample = 0;
        if (endSample < 0) endSample = 0;
        if (endSample >= totalSamples) endSample = 0;

        // Extract audio on main thread
        int length = (endSample >= startSample) ? (endSample - startSample) : ((totalSamples - startSample) + endSample);
        int maxLen = MaxAudioSeconds * SampleRate;
        if (length > maxLen) length = maxLen;

        if (length <= 0) { isTranscribing = false; isThinking = false; yield break; }

        EnsureFloatBuffer(ref _speechDataBuffer, length);
        try
        {
            if (endSample >= startSample)
            {
                micClip.GetData(_speechDataBuffer, startSample);
            }
            else
            {
                int part1Length = totalSamples - startSample;
                EnsureFloatBuffer(ref _wrapReadBuffer, Mathf.Max(part1Length, endSample));
                micClip.GetData(_wrapReadBuffer, startSample);
                System.Array.Copy(_wrapReadBuffer, 0, _speechDataBuffer, 0, part1Length);
                if (endSample > 0)
                {
                    micClip.GetData(_wrapReadBuffer, 0);
                    System.Array.Copy(_wrapReadBuffer, 0, _speechDataBuffer, part1Length, endSample);
                }
            }
        }
        catch
        {
            isTranscribing = false;
            isThinking = false;
            yield break;
        }

        // Single-pass normalization: find max and check volume at once
        float maxVol = 0f;
        for (int i = 0; i < length; i++)
        {
            float absValue = Mathf.Abs(_speechDataBuffer[i]);
            if (absValue > maxVol) maxVol = absValue;
        }
        if (maxVol < 0.01f) { isTranscribing = false; isThinking = false; yield break; }

        // Normalize in-place
        float scale = 1.0f / Mathf.Max(maxVol, 0.1f);
        for (int i = 0; i < length; i++) _speechDataBuffer[i] *= scale;

        // Pad buffer (only clear changed region)
        if (_lastPaddedLength > length)
        {
            System.Array.Clear(_paddedAudioBuffer, length, _lastPaddedLength - length);
        }
        _lastPaddedLength = length;
        System.Array.Copy(_speechDataBuffer, _paddedAudioBuffer, length);

        // Yield to let cube render before heavy encoding work starts
        yield return null;

        // Run encoder pipeline (stays on main thread for Sentis GPU safety)
        Tensor<float> encodedAudioCPU = null;
        int paddedLen = _paddedAudioBuffer.Length;
        {
            using Tensor<float> audioInput = new Tensor<float>(new TensorShape(1, paddedLen), _paddedAudioBuffer);
            spectroEngine.Schedule(audioInput);
            using (var logMel = spectroEngine.PeekOutput() as Tensor<float>)
            {
                encoderEngine.Schedule(logMel);
                using (var gpuOutput = encoderEngine.PeekOutput() as Tensor<float>)
                {
                    using (var nativeData = gpuOutput.DownloadToNativeArray())
                    {
                        encodedAudioCPU = new Tensor<float>(gpuOutput.shape, nativeData);
                    }
                }
            }
        }

        // Yield again before decoder to let cube render
        yield return null;

        yield return DecodeLoop(encodedAudioCPU);

        encodedAudioCPU.Dispose();
        isTranscribing = false;
        isThinking = false;
        }
    }

    IEnumerator DecodeLoop(Tensor<float> encodedAudio)
    {
        using (DecodeMarker.Auto())
        {
        // Only clear the portion we'll write to (3 initial tokens + decoder iterations)
        int maxClear = Mathf.Min(3 + maxTokens, outputTokens.Length);
        for (int i = 0; i < maxClear; i++) outputTokens[i] = 0;
        _transcriptBuilder.Clear();

        outputTokens[0] = START_OF_TRANSCRIPT;
        outputTokens[1] = useHebrew ? HEBREW : ENGLISH; 
        outputTokens[2] = TRANSCRIBE;
        int tokenCount = 3;
        
        singleTokenArray[0] = NO_TIME_STAMPS;

        int consecutiveRepeats = 0;
        int previousToken = -1;
        int decodeIterations = Mathf.Max(1, maxTokens);
        _recentTokens.Clear();

        for (int i = 0; i < decodeIterations && tokenCount < outputTokens.Length; i++)
        {
            using (var currentInput = new NativeArray<int>(tokenCount, Allocator.Temp))
            {
                NativeArray<int>.Copy(outputTokens, 0, currentInput, 0, tokenCount);
                using (var tokensTensor = new Tensor<int>(new TensorShape(1, tokenCount), currentInput))
                {
                    decoderEngine1.SetInput("input_ids", tokensTensor);
                    decoderEngine1.SetInput("encoder_hidden_states", encodedAudio);
                    decoderEngine1.Schedule();
                }
            }

            _kvTensorsToDispose.Clear();
            PassOutputsToInputs(decoderEngine1, decoderEngine2, _kvTensorsToDispose);
            try 
            {
                using (var lastTokenTensor = new Tensor<int>(new TensorShape(1, 1), singleTokenArray))
                {
                    decoderEngine2.SetInput("input_ids", lastTokenTensor);
                    decoderEngine2.Schedule();
                }
            }
            finally
            {
                for (int tensorIndex = 0; tensorIndex < _kvTensorsToDispose.Count; tensorIndex++)
                {
                    _kvTensorsToDispose[tensorIndex].Dispose();
                }
            }

            int nextToken = 0;
            
            using (var logitsTensor = decoderEngine2.PeekOutput("logits") as Tensor<float>)
            {
                argmaxEngine.Schedule(logitsTensor);
                using (var argmaxOutput = argmaxEngine.PeekOutput() as Tensor<int>)
                {
                    using (var nativeTokenData = argmaxOutput.DownloadToNativeArray())
                    {
                        nextToken = nativeTokenData[0];
                    }
                }
                
                if (i == 0)
                {
                    // Hallucination check logic for first token would involve reading logits
                }
            }

            if (nextToken == END_OF_TEXT) break;

            // Strict repetition filtering: break on any token repeat to prevent hallucinations
            if (_recentTokens.Contains(nextToken) || nextToken == previousToken)
            {
                consecutiveRepeats++;
                if (consecutiveRepeats >= 1) break;  // Break immediately on any repeat
            }
            else
            {
                consecutiveRepeats = 0;
            }
            previousToken = nextToken;
            _recentTokens.Add(nextToken);

            // Cache token lookup to avoid repeated array access
            if (nextToken < _tokensLength && tokens[nextToken] != null)
            {
                string word = SanitizeToBasicCharacters(GetUnicodeText(tokens[nextToken]));
                if (word.Length > 0)  // Faster than string.IsNullOrEmpty
                {
                    _transcriptBuilder.Append(word);
                }
            }

            outputTokens[tokenCount] = nextToken; 
            tokenCount++;
            singleTokenArray[0] = nextToken;

            yield return null; 
        }

        fullText = _transcriptBuilder.ToString();

        // Post-process: remove duplicate consecutive words
        fullText = RemoveDuplicateWords(fullText);

        string cleanUserText = CleanString(fullText);
        
        // Reject if result is too short (likely incomplete recognition)
        if (cleanUserText.Length < 3)
        {
            Debug.LogWarning($"[Recognition] Rejected: too short - '{fullText}'");
            isTranscribing = false;
            isThinking = false;
            fullText = "";
            yield break;
        }

        if (!string.IsNullOrEmpty(cleanUserText) && _currentSentences != null)
        {
            FindBestMatch(cleanUserText);
        }
        else
        {
            Debug.Log($"FINAL TRANSCRIPT: {fullText}");
        }
        }
    }

    void FindBestMatch(string userText)
    {
        string bestSentence = "";
        int lowestDistance = 1000;
        
        // Use pre-cleaned sentences to avoid redundant CleanString calls
        for (int i = 0; i < _cleanedSentences.Length; i++)
        {
            int distance = LevenshteinDistance(userText, _cleanedSentences[i]);
            
            if (distance < lowestDistance)
            {
                lowestDistance = distance;
                bestSentence = _currentSentences[i];
                
                // Early exit on perfect match
                if (distance == 0) break;
            }
        }

        int threshold = Mathf.Max(4, bestSentence.Length / 2);
        if (matchTolerance < 0.5f) threshold = (int)(bestSentence.Length * matchTolerance);

        if (lowestDistance <= threshold)
        {
            fullText = bestSentence;
            Debug.Log($"<color=green>MATCH: {bestSentence}</color>");
        }
        else
        {
            fullText = $"unmatched: ({fullText})";
            Debug.Log($"<color=red>FAIL. Closest: {bestSentence} (Dist: {lowestDistance})</color>");
        }
    }

    string CleanString(string input)
    {
        _sb.Clear();
        foreach (char c in input.ToLower())
        {
            if (char.IsLetterOrDigit(c) || char.IsWhiteSpace(c))
                _sb.Append(c);
        }
        return _sb.ToString().Trim();
    }

    string RemoveDuplicateWords(string input)
    {
        _sb.Clear();
        var words = input.Split(new[] { ' ', '\t', '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
        
        if (words.Length == 0) return "";
        if (words.Length == 1) return input.Trim();
        
        string lastWord = words[0].ToLower();
        _sb.Append(words[0]);
        
        for (int i = 1; i < words.Length; i++)
        {
            string word = words[i].ToLower();
            // Skip if identical to previous word (case-insensitive)
            if (word != lastWord)
            {
                _sb.Append(" ").Append(words[i]);
                lastWord = word;
            }
        }
        
        return _sb.ToString();
    }

    int LevenshteinDistance(string s, string t)
    {
        int n = s.Length;
        int m = t.Length;
        if (n == 0) return m;
        if (m == 0) return n;

        EnsureLevenshteinCapacity(m + 1);
        
        // Initialize first row
        for (int j = 0; j <= m; j++) _levenshteinPrev[j] = j;

        // DP: iterate rows avoiding string allocations
        for (int i = 1; i <= n; i++)
        {
            _levenshteinCurr[0] = i;
            char sChar = s[i - 1];  // Cache string access
            
            for (int j = 1; j <= m; j++)
            {
                int cost = (t[j - 1] == sChar) ? 0 : 1;  // Use cached char
                int deletion = _levenshteinPrev[j] + 1;
                int insertion = _levenshteinCurr[j - 1] + 1;
                int substitution = _levenshteinPrev[j - 1] + cost;
                // Use inline min to avoid Mathf.Min calls
                _levenshteinCurr[j] = deletion < insertion ? (deletion < substitution ? deletion : substitution) : (insertion < substitution ? insertion : substitution);
            }

            var tmp = _levenshteinPrev;
            _levenshteinPrev = _levenshteinCurr;
            _levenshteinCurr = tmp;
        }

        return _levenshteinPrev[m];
    }

    void PassOutputsToInputs(Worker source, Worker dest, List<Tensor> tensorsToDispose)
    {
        for(int layer=0; layer<4; layer++)
        {
            var kDec = source.PeekOutput($"present.{layer}.decoder.key") as Tensor<float>;
            var vDec = source.PeekOutput($"present.{layer}.decoder.value") as Tensor<float>;
            var kEnc = source.PeekOutput($"present.{layer}.encoder.key") as Tensor<float>;
            var vEnc = source.PeekOutput($"present.{layer}.encoder.value") as Tensor<float>;

            tensorsToDispose.Add(kDec); tensorsToDispose.Add(vDec);
            tensorsToDispose.Add(kEnc); tensorsToDispose.Add(vEnc);

            dest.SetInput($"past_key_values.{layer}.decoder.key", kDec);
            dest.SetInput($"past_key_values.{layer}.decoder.value", vDec);
            dest.SetInput($"past_key_values.{layer}.encoder.key", kEnc);
            dest.SetInput($"past_key_values.{layer}.encoder.value", vEnc);
        }
    }

    void SetupEngines()
    {
        OnDestroy();
        encoderEngine = new Worker(ModelLoader.Load(audioEncoder), BackendType.GPUCompute);
        decoderEngine1 = new Worker(ModelLoader.Load(audioDecoder1), BackendType.GPUCompute);
        decoderEngine2 = new Worker(ModelLoader.Load(audioDecoder2), BackendType.GPUCompute);
        spectroEngine = new Worker(ModelLoader.Load(logMelSpectro), BackendType.GPUCompute);
        
        FunctionalGraph graph = new FunctionalGraph();
        var input = graph.AddInput(DataType.Float, new DynamicTensorShape(1, 1, 51865));
        var amax = Functional.ArgMax(input, -1, false);
        argmaxEngine = new Worker(graph.Compile(amax), BackendType.GPUCompute);
    }

    void LoadVocab()
    {
        var vocab = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabAsset.text);
        tokens = new string[vocab.Count];
        foreach (var item in vocab) tokens[item.Value] = item.Key;
        _tokensLength = tokens.Length;  // Cache length
    }

    void SetupWhiteSpaceShifts()
    {
        for (int i = 0, n = 0; i < 256; i++) if (IsWhiteSpace((char)i)) whiteSpaceCharacters[n++] = i;
    }

    void SetupBasicSymbolsLookup()
    {
        // Pre-build lookup table for O(1) symbol checking
        for (int i = 0; i < BasicSymbols.Length; i++)
        {
            char c = BasicSymbols[i];
            if (c < 256) _basicSymbolsLookup[c] = true;
        }
    }

    bool IsWhiteSpace(char c) => !(('!' <= c && c <= '~') || ('¡' <= c && c <= '¬') || ('®' <= c && c <= 'ÿ'));

    string GetUnicodeText(string text)
    {
        var bytes = Encoding.GetEncoding("ISO-8859-1").GetBytes(ShiftCharacterDown(text));
        return Encoding.UTF8.GetString(bytes);
    }

    string ShiftCharacterDown(string text)
    {
        // Avoid StringBuilder allocation for small operations - build directly
        char[] result = new char[text.Length];
        for (int i = 0; i < text.Length; i++)
        {
            char letter = text[i];
            result[i] = ((int)letter <= 256) ? letter : (char)whiteSpaceCharacters[(int)(letter - 256)];
        }
        return new string(result);
    }

    string SanitizeToBasicCharacters(string input)
    {
        _sanitizeBuilder.Clear();
        for (int i = 0; i < input.Length; i++)
        {
            char c = input[i];
            // Use lookup table for ASCII symbols (O(1) instead of O(n) for IndexOf)
            if (char.IsLetterOrDigit(c) || char.IsWhiteSpace(c) || (c < 256 && _basicSymbolsLookup[c]))
            {
                _sanitizeBuilder.Append(c);
            }
        }
        return _sanitizeBuilder.ToString();
    }

    void EnsureFloatBuffer(ref float[] buffer, int requiredLength)
    {
        if (buffer == null || buffer.Length < requiredLength)
        {
            buffer = new float[requiredLength];
        }
    }

    void EnsureLevenshteinCapacity(int requiredLength)
    {
        if (_levenshteinPrev == null || _levenshteinPrev.Length < requiredLength)
        {
            _levenshteinPrev = new int[requiredLength];
            _levenshteinCurr = new int[requiredLength];
        }
    }

    void OnDestroy()
    {
        encoderEngine?.Dispose(); decoderEngine1?.Dispose();
        decoderEngine2?.Dispose(); spectroEngine?.Dispose(); argmaxEngine?.Dispose();
        
        if (outputTokens.IsCreated) outputTokens.Dispose();
        if (singleTokenArray.IsCreated) singleTokenArray.Dispose();
    }
}
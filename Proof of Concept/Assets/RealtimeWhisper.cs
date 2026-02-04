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
    public float volumeThreshold = 0.02f;   
    public float requiredSilence = 0.6f;    

    [Header("Languages & Context")]
    public bool useHebrew = false; 
    public string[] englishSentences; 
    public string[] hebrewSentences; 
    
    [Tooltip("0.0 = Exact match required. 0.4 = Allows loose matching.")]
    [Range(0.0f, 1.0f)]
    public float matchTolerance = 0.3f; 
    
    [Header("Generation Settings")]
    public int maxTokens = 30;
    
    [Header("Anti-Hallucination")]
    public float minConfidence = 10.0f; 
    public string[] bannedPhrases = new string[] { "You", "Thank you", "MBC", "Copyright", "Subtitles", "watching" };

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
    private StringBuilder _sb = new StringBuilder(); 

    // Workers
    private Worker encoderEngine, decoderEngine1, decoderEngine2, spectroEngine, argmaxEngine;
    
    // Data
    private NativeArray<int> outputTokens; 
    private NativeArray<int> singleTokenArray; 
    
    private AudioClip micClip;
    private string micDevice;
    private const int SampleRate = 16000; 
    private bool isTranscribing = false;

    // Constants
    const int END_OF_TEXT = 50257;
    const int START_OF_TRANSCRIPT = 50258;
    const int ENGLISH = 50259;
    const int HEBREW = 50279; 
    const int TRANSCRIBE = 50359;
    const int NO_TIME_STAMPS = 50363;

    private string[] tokens;
    private int[] whiteSpaceCharacters = new int[256];

    void Start()
    {
        SetupWhiteSpaceShifts();
        LoadVocab();
        SetupEngines();
        SetupMicrophone();
        
        outputTokens = new NativeArray<int>(100, Allocator.Persistent);
        singleTokenArray = new NativeArray<int>(1, Allocator.Persistent);
        singleTokenArray[0] = NO_TIME_STAMPS;

        SetLanguage(useHebrew);
    }

    public void SetLanguage(bool isHebrew)
    {
        useHebrew = isHebrew;
        _currentSentences = isHebrew ? hebrewSentences : englishSentences;
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
        if (isTranscribing) return;

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

    float GetCurrentVolume()
    {
        int sampleWindow = 128; 
        int micPos = Microphone.GetPosition(micDevice);
        if (micPos < sampleWindow + 1) return 0f;
        float[] waveData = new float[sampleWindow];
        micClip.GetData(waveData, micPos - sampleWindow);
        float max = 0f;
        foreach (var v in waveData) if (Mathf.Abs(v) > max) max = Mathf.Abs(v);
        return max;
    }

    IEnumerator TranscribeRoutine(int startSample, int endSample)
    {
        isTranscribing = true;
        isThinking = true;
        fullText = ""; 

        int totalSamples = micClip.samples;
        if (startSample < 0) startSample = 0;
        if (startSample >= totalSamples) startSample = 0;
        if (endSample < 0) endSample = 0;
        if (endSample >= totalSamples) endSample = 0;

        int length = (endSample >= startSample) ? (endSample - startSample) : ((totalSamples - startSample) + endSample);
        int maxLen = 30 * SampleRate; 
        if (length > maxLen) length = maxLen;

        if (length <= 0) { isTranscribing = false; isThinking = false; yield break; }

        float[] speechData = new float[length];
        bool readSuccess = true;
        try
        {
            if (endSample >= startSample)
            {
                micClip.GetData(speechData, startSample);
            }
            else
            {
                float[] part1 = new float[totalSamples - startSample];
                micClip.GetData(part1, startSample);
                float[] part2 = new float[endSample];
                micClip.GetData(part2, 0);
                System.Array.Copy(part1, 0, speechData, 0, part1.Length);
                System.Array.Copy(part2, 0, speechData, part1.Length, part2.Length);
            }
        }
        catch (System.Exception) { readSuccess = false; }

        if (!readSuccess) { isTranscribing = false; isThinking = false; yield break; }

        float maxVol = 0f;
        foreach(var v in speechData) if(Mathf.Abs(v) > maxVol) maxVol = Mathf.Abs(v);
        if (maxVol < 0.01f) { isTranscribing = false; isThinking = false; yield break; }

        float scale = 1.0f / Mathf.Max(maxVol, 0.1f); 
        for(int i=0; i<speechData.Length; i++) speechData[i] *= scale;

        float[] paddedBuffer = new float[maxLen];
        System.Array.Copy(speechData, paddedBuffer, speechData.Length); 

        Tensor<float> encodedAudioCPU = null;

        {
            using Tensor<float> audioInput = new Tensor<float>(new TensorShape(1, paddedBuffer.Length), paddedBuffer);
            spectroEngine.Schedule(audioInput);
            using (var logMel = spectroEngine.PeekOutput() as Tensor<float>)
            {
                encoderEngine.Schedule(logMel);
                using (var gpuOutput = encoderEngine.PeekOutput() as Tensor<float>)
                {
                    // FIX: Use DownloadToNativeArray instead of ReadbackAndClone to avoid Awaitable/Blocking confusion
                    using (var nativeData = gpuOutput.DownloadToNativeArray())
                    {
                        // Create a CPU tensor copy from the native data
                        encodedAudioCPU = new Tensor<float>(gpuOutput.shape, nativeData);
                    }
                }
            }
        } 

        yield return DecodeLoop(encodedAudioCPU);

        encodedAudioCPU.Dispose(); 
        isTranscribing = false;
        isThinking = false;
    }

    IEnumerator DecodeLoop(Tensor<float> encodedAudio)
    {
        for (int i = 0; i < outputTokens.Length; i++) outputTokens[i] = 0;

        outputTokens[0] = START_OF_TRANSCRIPT;
        outputTokens[1] = useHebrew ? HEBREW : ENGLISH; 
        outputTokens[2] = TRANSCRIBE;
        int tokenCount = 3;
        
        singleTokenArray[0] = NO_TIME_STAMPS;

        int consecutiveRepeats = 0;
        int previousToken = -1;

        for (int i = 0; i < maxTokens; i++) 
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

            List<Tensor> linkTensors = PassOutputsToInputs(decoderEngine1, decoderEngine2);
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
                foreach (var t in linkTensors) t.Dispose();
            }

            int nextToken = 0;
            
            using (var logitsTensor = decoderEngine2.PeekOutput("logits") as Tensor<float>)
            {
                argmaxEngine.Schedule(logitsTensor);
                using (var argmaxOutput = argmaxEngine.PeekOutput() as Tensor<int>)
                {
                    // FIX: Use DownloadToNativeArray for immediate blocking read
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

            if (nextToken == previousToken)
            {
                consecutiveRepeats++;
                if (consecutiveRepeats >= 2) break;
            }
            else
            {
                consecutiveRepeats = 0;
            }
            previousToken = nextToken;

            if (nextToken < tokens.Length)
            {
                string word = GetUnicodeText(tokens[nextToken]);
                fullText += word;
            }

            outputTokens[tokenCount] = nextToken; 
            tokenCount++;
            singleTokenArray[0] = nextToken;

            yield return null; 
        }

        string cleanUserText = CleanString(fullText);
        
        foreach(var badPhrase in bannedPhrases)
        {
            if (cleanUserText.Contains(CleanString(badPhrase))) 
            {
                Debug.Log($"Hallucination detected: {fullText}");
                cleanUserText = ""; 
                fullText = "";
            }
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

    void FindBestMatch(string userText)
    {
        string bestSentence = "";
        int lowestDistance = 1000;
        
        foreach (var sentence in _currentSentences)
        {
            string cleanTarget = CleanString(sentence);
            int distance = LevenshteinDistance(userText, cleanTarget);
            
            if (distance < lowestDistance)
            {
                lowestDistance = distance;
                bestSentence = sentence;
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
            fullText = $"#unmatched# ({fullText})";
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

    public static int LevenshteinDistance(string s, string t)
    {
        int n = s.Length;
        int m = t.Length;
        int[,] d = new int[n + 1, m + 1];
        if (n == 0) return m;
        if (m == 0) return n;
        for (int i = 0; i <= n; d[i, 0] = i++) { }
        for (int j = 0; j <= m; d[0, j] = j++) { }
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
            {
                int cost = (t[j - 1] == s[i - 1]) ? 0 : 1;
                d[i, j] = Mathf.Min(Mathf.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + cost);
            }
        return d[n, m];
    }

    List<Tensor> PassOutputsToInputs(Worker source, Worker dest)
    {
        var tensorsToDispose = new List<Tensor>();
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
        return tensorsToDispose;
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
    }

    void SetupWhiteSpaceShifts()
    {
        for (int i = 0, n = 0; i < 256; i++) if (IsWhiteSpace((char)i)) whiteSpaceCharacters[n++] = i;
    }

    bool IsWhiteSpace(char c) => !(('!' <= c && c <= '~') || ('¡' <= c && c <= '¬') || ('®' <= c && c <= 'ÿ'));

    string GetUnicodeText(string text)
    {
        var bytes = Encoding.GetEncoding("ISO-8859-1").GetBytes(ShiftCharacterDown(text));
        return Encoding.UTF8.GetString(bytes);
    }

    string ShiftCharacterDown(string text)
    {
        string outText = "";
        foreach (char letter in text) outText += ((int)letter <= 256) ? letter : (char)whiteSpaceCharacters[(int)(letter - 256)];
        return outText;
    }

    void OnDestroy()
    {
        encoderEngine?.Dispose(); decoderEngine1?.Dispose();
        decoderEngine2?.Dispose(); spectroEngine?.Dispose(); argmaxEngine?.Dispose();
        
        if (outputTokens.IsCreated) outputTokens.Dispose();
        if (singleTokenArray.IsCreated) singleTokenArray.Dispose();
    }
}
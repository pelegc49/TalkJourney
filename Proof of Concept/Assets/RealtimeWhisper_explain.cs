using UnityEngine; // Unity core engine functionality
using UnityEngine.UI; // Unity UI elements (though not strictly used in this script logic directly, often standard)
using Unity.InferenceEngine; // Namespace for the AI model inference (Sentis 2.0+)
using System.Collections; // For Coroutines (IEnumerator)
using System.Collections.Generic; // For Lists and Dictionaries
using System.Text; // For StringBuilder
using Unity.Collections; // For NativeArray (efficient memory management for Unity jobs/engine)
using Newtonsoft.Json; // For parsing the vocab.json file
using System.Linq; // For LINQ queries (if used)
using System.Text.RegularExpressions;  // For regex based text processing (stutter removal)

#if ENABLE_INPUT_SYSTEM 
using UnityEngine.InputSystem; // If using new Input System, import its namespace
#endif

public class RealtimeWhisper_explain : MonoBehaviour // Main class inheriting from MonoBehaviour to run in Unity
{
    [Header("Model Assets")] // Inspector header for model references
    public ModelAsset audioDecoder1; // Reference to the first decoder ONNX model
    public ModelAsset audioDecoder2; // Reference to the second decoder ONNX model (with past key values)
    public ModelAsset audioEncoder; // Reference to the encoder ONNX model
    public ModelAsset logMelSpectro; // Reference to the spectrogram pre-processing model
    public TextAsset vocabAsset; // Reference to the vocabulary JSON file

    [Header("Microphone Settings")] // Inspector header for mic settings
    public int recordingDuration = 30;  // Duration of the audio buffer (Whisper needs 30s)
    public KeyCode triggerKey = KeyCode.Space; // Key to trigger transcription manually

    [Header("VAD Settings")] // Inspector header for Voice Activity Detection
    [Range(0.0f, 1.0f)] // Slider range for threshold
    public float volumeThreshold = 0.02f;    // Volume level required to trigger speech detection
    public float requiredSilence = 0.6f;     // Time in seconds of silence before stopping recording

    [Header("Languages & Context")] // Inspector header for language and context
    public bool useHebrew = false; // Toggle to switch between English and Hebrew
    public string[] englishSentences; // List of allowed English sentences for context matching
    public string[] hebrewSentences; // List of allowed Hebrew sentences for context matching
    
    [Tooltip("0.0 = Exact match required. 0.4 = Allows loose matching.")] // Tooltip for tolerance
    [Range(0.0f, 1.0f)] // Slider range
    public float matchTolerance = 0.3f;  // Levenshtein distance tolerance ratio
    
    [Header("Generation Settings")] // Inspector header for generation parameters
    public int maxTokens = 30; // Maximum number of tokens to generate per inference
    
    [Header("Anti-Hallucination")] // Inspector header for hallucination prevention
    public float minConfidence = 10.0f;  // Minimum logit score for the first token to accept it
    public string[] bannedPhrases = new string[] { "You", "Thank you", "MBC", "Copyright", "Subtitles", "watching" }; // Common Whisper hallucinations to filter out

    [Header("Output")] // Inspector header for output visualization
    public string fullText = "";  // The final transcribed text displayed in Inspector
    public bool isThinking = false; // Flag to indicate if inference is currently running

    // Internal State
    private float silenceTimer = 0f; // Timer to track duration of silence
    private bool isSpeaking = false; // Flag tracking if user is currently speaking
    private int vadStartPos = 0; // Sample index where speech started
    private int vadEndPos = 0; // Sample index where speech ended
    
    // Current active context
    private string[] _currentSentences; // Reference to currently active language sentence list
    private StringBuilder _sb = new StringBuilder();  // StringBuilder for efficient string manipulation

    // Workers
    private Worker encoderEngine, decoderEngine1, decoderEngine2, spectroEngine, argmaxEngine; // Workers run the neural network models
    
    // Data
    private NativeArray<int> outputTokens;  // Array to store generated tokens during inference
    private NativeArray<int> singleTokenArray;  // Array to hold a single token for input
    
    private AudioClip micClip; // Reference to the microphone audio clip
    private string micDevice; // Name of the microphone device
    private const int SampleRate = 16000;  // Whisper requires exactly 16kHz sample rate
    private bool isTranscribing = false; // Flag to prevent overlapping transcriptions

    // Constants
    const int END_OF_TEXT = 50257; // Token ID representing end of text
    const int START_OF_TRANSCRIPT = 50258; // Token ID representing start of transcript
    const int ENGLISH = 50259; // Token ID for English language
    const int HEBREW = 50279; // Token ID for Hebrew language (Updated from user request)
    const int TRANSCRIBE = 50359; // Token ID for transcribe task
    const int NO_TIME_STAMPS = 50363; // Token ID to suppress timestamp generation

    private string[] tokens; // Array holding the string representation of all tokens (vocabulary)
    private int[] whiteSpaceCharacters = new int[256]; // Mapping for special whitespace character handling

    void Start() // Unity Start method
    {
        SetupWhiteSpaceShifts(); // Initialize whitespace mapping
        LoadVocab(); // Load vocabulary from JSON
        SetupEngines(); // Initialize AI models and workers
        SetupMicrophone(); // Initialize microphone recording
        
        outputTokens = new NativeArray<int>(100, Allocator.Persistent); // Allocate memory for token output history
        singleTokenArray = new NativeArray<int>(1, Allocator.Persistent); // Allocate memory for single token input
        singleTokenArray[0] = NO_TIME_STAMPS; // Initialize single token array default value

        // Set initial language context
        SetLanguage(useHebrew); // Set language based on inspector toggle
    }

    // --- NEW: Language Switching Logic ---
    public void SetLanguage(bool isHebrew) // Method to switch language context
    {
        useHebrew = isHebrew; // Update internal boolean
        _currentSentences = isHebrew ? hebrewSentences : englishSentences; // Point current sentences to correct language array
        
        // We don't need to rebuild grammar masks for the simple distance check,
        // but if you want strict boosting later, you'd update it here.
        Debug.Log($"Language switched to: {(useHebrew ? "Hebrew" : "English")}"); // Log change
    }

    // Inspector validator to update context at runtime
    private void OnValidate() // Unity Editor method called when values change in Inspector
    {
        if (Application.isPlaying) // Only run logic if game is playing
        {
            SetLanguage(useHebrew); // Apply language change immediately
        }
    }

    void SetupMicrophone() // Initialize microphone
    {
        if (Microphone.devices.Length > 0) // Check if any mic exists
        {
            micDevice = Microphone.devices[0]; // Use default device
            micClip = Microphone.Start(micDevice, true, recordingDuration, SampleRate); // Start continuous recording
            Debug.Log($"Microphone Initialized: {micDevice} @ {SampleRate}Hz"); // Log success
        }
        else // No mic found
        {
            Debug.LogError("No Microphone detected!"); // Log error
        }
    }

    void Update() // Unity Update loop (per frame)
    {
        if (isTranscribing) return; // If already processing, skip logic

        bool manualTrigger = false; // Flag for manual key press
        #if ENABLE_INPUT_SYSTEM 
            if (Keyboard.current != null && Keyboard.current.spaceKey.wasPressedThisFrame) manualTrigger = true; // Check new input system Space key
        #else
            if (Input.GetKeyDown(triggerKey)) manualTrigger = true; // Check legacy input system trigger key
        #endif

        if (manualTrigger) // If manual trigger pressed
        {
            int now = Microphone.GetPosition(micDevice); // Get current mic position
            int start = now - (recordingDuration * SampleRate);  // Calculate start position (backtracking duration)
            if (start < 0) start += micClip.samples; // Handle wrap-around
            StartCoroutine(TranscribeRoutine(start, now)); // Start transcription for specific segment
            return; // Exit update
        }

        float currentVol = GetCurrentVolume(); // Measure current volume

        if (currentVol > volumeThreshold) // If volume exceeds threshold (Speech detected)
        {
            if (!isSpeaking)  // If not previously speaking
            {
                isSpeaking = true; // Set speaking flag
                int frequency = micClip.frequency; // Get mic frequency
                int offset = (int)(0.4f * frequency);  // Calculate 0.4s buffer offset
                vadStartPos = Microphone.GetPosition(micDevice) - offset; // Set start position with buffer
                if (vadStartPos < 0) vadStartPos += micClip.samples; // Handle wrap-around
            }
            silenceTimer = 0f;  // Reset silence timer
        }
        else if (isSpeaking) // If currently speaking but volume is low
        {
            silenceTimer += Time.deltaTime; // Increment silence timer
            if (silenceTimer > requiredSilence) // If silence exceeds required duration
            {
                isSpeaking = false; // Stop speaking flag
                vadEndPos = Microphone.GetPosition(micDevice); // Set end position
                StartCoroutine(TranscribeRoutine(vadStartPos, vadEndPos)); // Start transcription for detected segment
            }
        }
    }

    float GetCurrentVolume() // Calculate RMS volume of recent audio
    {
        int sampleWindow = 128;  // Window size for analysis
        int micPos = Microphone.GetPosition(micDevice); // Get current position
        if (micPos < sampleWindow + 1) return 0f; // Check bounds
        float[] waveData = new float[sampleWindow]; // Allocate temp array
        micClip.GetData(waveData, micPos - sampleWindow); // Get recent data
        float max = 0f; // Initialize max
        foreach (var v in waveData) if (Mathf.Abs(v) > max) max = Mathf.Abs(v); // Find max amplitude
        return max; // Return volume
    }

    IEnumerator TranscribeRoutine(int startSample, int endSample) // Coroutine handling audio processing and inference start
    {
        isTranscribing = true; // Lock transcription
        isThinking = true; // Set UI flag
        fullText = "";  // Clear previous text

        int totalSamples = micClip.samples; // Get total buffer length
        if (startSample < 0) startSample = 0; // Safety clamp
        if (startSample >= totalSamples) startSample = 0; // Safety clamp
        if (endSample < 0) endSample = 0; // Safety clamp
        if (endSample >= totalSamples) endSample = 0; // Safety clamp

        // Calculate length of segment accounting for wrap-around
        int length = (endSample >= startSample) ? (endSample - startSample) : ((totalSamples - startSample) + endSample);
        int maxLen = 30 * SampleRate;  // Max Whisper input length (30s)
        if (length > maxLen) length = maxLen; // Clamp length to max

        if (length <= 0) { isTranscribing = false; isThinking = false; yield break; } // Exit if empty

        float[] speechData = new float[length]; // Allocate array for speech data
        bool readSuccess = true; // Flag for read success
        try
        {
            if (endSample >= startSample) // Standard case (no wrap)
            {
                micClip.GetData(speechData, startSample); // Get data directly
            }
            else // Wrap-around case
            {
                float[] part1 = new float[totalSamples - startSample]; // Allocate part 1
                micClip.GetData(part1, startSample); // Get data to end of buffer
                float[] part2 = new float[endSample]; // Allocate part 2
                micClip.GetData(part2, 0); // Get data from start of buffer
                System.Array.Copy(part1, 0, speechData, 0, part1.Length); // Copy part 1
                System.Array.Copy(part2, 0, speechData, part1.Length, part2.Length); // Copy part 2
            }
        }
        catch (System.Exception) { readSuccess = false; } // Catch potential audio engine errors

        if (!readSuccess) { isTranscribing = false; isThinking = false; yield break; } // Exit on error

        float maxVol = 0f; // Calculate max volume for normalization
        foreach(var v in speechData) if(Mathf.Abs(v) > maxVol) maxVol = Mathf.Abs(v);
        if (maxVol < 0.01f) { isTranscribing = false; isThinking = false; yield break; } // Exit if silent

        float scale = 1.0f / Mathf.Max(maxVol, 0.1f);  // Calculate normalization scale
        for(int i=0; i<speechData.Length; i++) speechData[i] *= scale; // Normalize audio

        float[] paddedBuffer = new float[maxLen]; // Create 30s padded buffer
        System.Array.Copy(speechData, paddedBuffer, speechData.Length);  // Copy speech into padded buffer

        Tensor<float> encodedAudioCPU = null; // Variable to hold encoder output on CPU

        {
            using Tensor<float> audioInput = new Tensor<float>(new TensorShape(1, paddedBuffer.Length), paddedBuffer); // Create input tensor
            spectroEngine.Schedule(audioInput); // Run spectrogram model
            using (var logMel = spectroEngine.PeekOutput() as Tensor<float>) // Get spectrogram output
            {
                encoderEngine.Schedule(logMel); // Run encoder model
                using (var gpuOutput = encoderEngine.PeekOutput() as Tensor<float>) // Get encoder output (GPU)
                {
                    encodedAudioCPU = gpuOutput.ReadbackAndClone() as Tensor<float>; // Download to CPU
                }
            }
        } 

        yield return DecodeLoop(encodedAudioCPU); // Run decoding loop

        encodedAudioCPU.Dispose();  // Dispose CPU tensor
        isTranscribing = false; // Unlock
        isThinking = false; // Reset flag
    }

    IEnumerator DecodeLoop(Tensor<float> encodedAudio) // Coroutine for auto-regressive decoding
    {
        for (int i = 0; i < outputTokens.Length; i++) outputTokens[i] = 0; // Clear token history

        outputTokens[0] = START_OF_TRANSCRIPT; // Set start token
        
        // --- Language Selection ---
        outputTokens[1] = useHebrew ? HEBREW : ENGLISH;  // Set language token
        outputTokens[2] = TRANSCRIBE; // Set task token
        int tokenCount = 3; // Initial token count
        
        singleTokenArray[0] = NO_TIME_STAMPS; // Set initial input for decoder 2

        int consecutiveRepeats = 0; // Counter for repeated words
        int previousToken = -1; // Track previous token

        for (int i = 0; i < maxTokens; i++)  // Generation loop
        {
            using (var currentInput = new NativeArray<int>(tokenCount, Allocator.Temp)) // Create temp array for history
            {
                NativeArray<int>.Copy(outputTokens, 0, currentInput, 0, tokenCount); // Copy valid tokens
                using (var tokensTensor = new Tensor<int>(new TensorShape(1, tokenCount), currentInput)) // Create tensor
                {
                    decoderEngine1.SetInput("input_ids", tokensTensor); // Set input
                    decoderEngine1.SetInput("encoder_hidden_states", encodedAudio); // Set encoder state
                    decoderEngine1.Schedule(); // Run decoder 1
                }
            }

            List<Tensor> linkTensors = PassOutputsToInputs(decoderEngine1, decoderEngine2); // Link KV cache
            try 
            {
                using (var lastTokenTensor = new Tensor<int>(new TensorShape(1, 1), singleTokenArray)) // Create input for decoder 2
                {
                    decoderEngine2.SetInput("input_ids", lastTokenTensor); // Set input
                    decoderEngine2.Schedule(); // Run decoder 2
                }
            }
            finally
            {
                foreach (var t in linkTensors) t.Dispose(); // Dispose intermediate tensors
            }

            int nextToken = 0; // Variable for result
            
            using (var logitsTensor = decoderEngine2.PeekOutput("logits") as Tensor<float>) // Get logits
            {
                argmaxEngine.Schedule(logitsTensor); // Run ArgMax model
                using (var argmaxOutput = argmaxEngine.PeekOutput() as Tensor<int>) // Get token ID tensor
                {
                    using (var cpuToken = argmaxOutput.ReadbackAndClone() as Tensor<int>) // Download to CPU
                    {
                        nextToken = cpuToken[0]; // Read value
                    }
                }
                
                // Basic hallucination check for first token logic goes here
            }

            if (nextToken == END_OF_TEXT) break; // Stop if end token

            // Repetition Guard
            if (nextToken == previousToken) // If token repeats
            {
                consecutiveRepeats++; // Increment count
                if (consecutiveRepeats >= 2) break; // Break if repeating too much
            }
            else
            {
                consecutiveRepeats = 0; // Reset count
            }
            previousToken = nextToken; // Update previous

            if (nextToken < tokens.Length) // Safety check
            {
                string word = GetUnicodeText(tokens[nextToken]); // Decode token to text
                fullText += word; // Append to output
            }

            outputTokens[tokenCount] = nextToken;  // Add to history
            tokenCount++; // Increment count
            singleTokenArray[0] = nextToken; // Set input for next step

            yield return null; // Yield frame
        }

        // --- 4. CLEANING & MATCHING (Inspired by reference code) ---
        string cleanUserText = CleanString(fullText); // Clean result string
        
        // Remove Hallucinations
        foreach(var badPhrase in bannedPhrases) // Iterate banned list
        {
            if (cleanUserText.Contains(CleanString(badPhrase)))  // Check if present
            {
                Debug.Log($"Hallucination detected: {fullText}"); // Log detection
                cleanUserText = "";  // Wipe text
                fullText = ""; // Wipe display text
            }
        }

        if (!string.IsNullOrEmpty(cleanUserText) && _currentSentences != null) // If text exists and context is set
        {
            FindBestMatch(cleanUserText); // Run matching logic
        }
        else
        {
            Debug.Log($"FINAL TRANSCRIPT: {fullText}"); // Log raw output
        }
    }

    // --- Inspired Matching Logic ---
    void FindBestMatch(string userText) // Find closest sentence
    {
        string bestSentence = ""; // Holder for best match
        int lowestDistance = 1000; // Initialize distance high
        
        foreach (var sentence in _currentSentences) // Loop through allowed sentences
        {
            string cleanTarget = CleanString(sentence); // Clean target sentence
            int distance = LevenshteinDistance(userText, cleanTarget); // Calc distance
            
            if (distance < lowestDistance) // If better match
            {
                lowestDistance = distance; // Update best distance
                bestSentence = sentence; // Update best sentence
            }
        }

        // Calculate Threshold
        int threshold = Mathf.Max(4, bestSentence.Length / 2); // Base threshold calculation
        // Override with strict tolerance if needed
        if (matchTolerance < 0.5f) threshold = (int)(bestSentence.Length * matchTolerance); // Apply strict tolerance

        if (lowestDistance <= threshold) // If within threshold
        {
            fullText = bestSentence; // Snap to correct sentence
            Debug.Log($"<color=green>MATCH: {bestSentence}</color>"); // Log match
        }
        else // If not match
        {
            fullText = $"#unmatched# ({fullText})"; // Set unmatched text
            Debug.Log($"<color=red>FAIL. Closest: {bestSentence} (Dist: {lowestDistance})</color>"); // Log failure
        }
    }

    string CleanString(string input) // Helper to clean strings
    {
        _sb.Clear(); // Clear builder
        foreach (char c in input.ToLower()) // Iterate lower case chars
        {
            if (char.IsLetterOrDigit(c) || char.IsWhiteSpace(c)) // Keep only alphanumeric and spaces
                _sb.Append(c); // Append valid char
        }
        return _sb.ToString().Trim(); // Return cleaned string
    }

    public static int LevenshteinDistance(string s, string t) // Standard Levenshtein Distance Algo
    {
        int n = s.Length; // Length s
        int m = t.Length; // Length t
        int[,] d = new int[n + 1, m + 1]; // Distance matrix
        if (n == 0) return m; // Edge case
        if (m == 0) return n; // Edge case
        for (int i = 0; i <= n; d[i, 0] = i++) { } // Initialize row
        for (int j = 0; j <= m; d[0, j] = j++) { } // Initialize col
        for (int i = 1; i <= n; i++) // Iterate rows
            for (int j = 1; j <= m; j++) // Iterate cols
            {
                int cost = (t[j - 1] == s[i - 1]) ? 0 : 1; // Calc match cost
                d[i, j] = Mathf.Min(Mathf.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + cost); // Min of insert, del, sub
            }
        return d[n, m]; // Return distance
    }

    // --- Standard Inference Helpers ---
    List<Tensor> PassOutputsToInputs(Worker source, Worker dest) // Helper to pass state between decoders
    {
        var tensorsToDispose = new List<Tensor>(); // List to track created tensors for disposal
        for(int layer=0; layer<4; layer++) // Iterate layers (Whisper Tiny has 4)
        {
            // Peek output tensors from source worker
            var kDec = source.PeekOutput($"present.{layer}.decoder.key") as Tensor<float>;
            var vDec = source.PeekOutput($"present.{layer}.decoder.value") as Tensor<float>;
            var kEnc = source.PeekOutput($"present.{layer}.encoder.key") as Tensor<float>;
            var vEnc = source.PeekOutput($"present.{layer}.encoder.value") as Tensor<float>;

            // Add to disposal list
            tensorsToDispose.Add(kDec); tensorsToDispose.Add(vDec);
            tensorsToDispose.Add(kEnc); tensorsToDispose.Add(vEnc);

            // Set as inputs for destination worker
            dest.SetInput($"past_key_values.{layer}.decoder.key", kDec);
            dest.SetInput($"past_key_values.{layer}.decoder.value", vDec);
            dest.SetInput($"past_key_values.{layer}.encoder.key", kEnc);
            dest.SetInput($"past_key_values.{layer}.encoder.value", vEnc);
        }
        return tensorsToDispose; // Return list
    }

    void SetupEngines() // Initialize neural networks
    {
        OnDestroy(); // Cleanup any existing engines first
        // Load models using ModelLoader
        encoderEngine = new Worker(ModelLoader.Load(audioEncoder), BackendType.GPUCompute);
        decoderEngine1 = new Worker(ModelLoader.Load(audioDecoder1), BackendType.GPUCompute);
        decoderEngine2 = new Worker(ModelLoader.Load(audioDecoder2), BackendType.GPUCompute);
        spectroEngine = new Worker(ModelLoader.Load(logMelSpectro), BackendType.GPUCompute);
        
        // Build ArgMax model graph
        FunctionalGraph graph = new FunctionalGraph();
        var input = graph.AddInput(DataType.Float, new DynamicTensorShape(1, 1, 51865));
        var amax = Functional.ArgMax(input, -1, false);
        argmaxEngine = new Worker(graph.Compile(amax), BackendType.GPUCompute);
    }

    void LoadVocab() // Load tokenizer vocabulary
    {
        var vocab = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabAsset.text); // Parse JSON
        tokens = new string[vocab.Count]; // Allocate array
        foreach (var item in vocab) tokens[item.Value] = item.Key; // Populate array by index
    }

    void SetupWhiteSpaceShifts() // Setup byte-level decoding map
    {
        for (int i = 0, n = 0; i < 256; i++) if (IsWhiteSpace((char)i)) whiteSpaceCharacters[n++] = i;
    }

    bool IsWhiteSpace(char c) => !(('!' <= c && c <= '~') || ('¡' <= c && c <= '¬') || ('®' <= c && c <= 'ÿ')); // Check BPE whitespace char

    string GetUnicodeText(string text) // Convert BPE token to string
    {
        var bytes = Encoding.GetEncoding("ISO-8859-1").GetBytes(ShiftCharacterDown(text)); // Get bytes using shifted chars
        return Encoding.UTF8.GetString(bytes); // Convert bytes to UTF8 string
    }

    string ShiftCharacterDown(string text) // Map special BPE chars back to standard bytes
    {
        string outText = "";
        foreach (char letter in text) outText += ((int)letter <= 256) ? letter : (char)whiteSpaceCharacters[(int)(letter - 256)];
        return outText;
    }

    void OnDestroy() // Cleanup resources on object destruction
    {
        // Dispose all workers
        encoderEngine?.Dispose(); decoderEngine1?.Dispose();
        decoderEngine2?.Dispose(); spectroEngine?.Dispose(); argmaxEngine?.Dispose();
        
        // Dispose persistent arrays
        if (outputTokens.IsCreated) outputTokens.Dispose();
        if (singleTokenArray.IsCreated) singleTokenArray.Dispose();
    }
}
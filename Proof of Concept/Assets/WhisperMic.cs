using UnityEngine;
using UnityEngine.UI;
using Unity.InferenceEngine; // The ONLY namespace you should use for AI
using System.Collections;

// REMOVED: using Unity.Barracuda; (This was causing the conflict!)

public class WhisperMic : MonoBehaviour
{
    // Assign these in the Inspector
    public ModelAsset audioDecoder1;
    public ModelAsset audioDecoder2;
    public ModelAsset audioEncoder;
    public ModelAsset logMelSpectro;
    public ModelAsset vocab;

    // Internal variables
    private AudioClip micClip;
    private string micDevice;
    private const int SampleRate = 16000; // Whisper expects exactly 16kHz
    private const int MaxDuration = 30;   // Whisper expects exactly 30s context
    
    // UPDATED: Use 'Worker' instead of 'IWorker'
    private Worker encoderEngine;
    private Worker decoderEngine1;
    private Worker decoderEngine2;
    private Worker spectroEngine;
    
    private bool isTranscribing = false;

    void Start()
    {
        // 1. Setup the Microphone
        if (Microphone.devices.Length > 0)
        {
            micDevice = Microphone.devices[0];
            // Start recording into a looping 30s buffer
            micClip = Microphone.Start(micDevice, true, MaxDuration, SampleRate);
        }
        else
        {
            Debug.LogError("No Microphone detected!");
        }

        // 2. Initialize Engines
        SetupEngines();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space) && !isTranscribing)
        {
            StartCoroutine(TranscribeMicData());
        }
    }

    private IEnumerator TranscribeMicData()
    {
        isTranscribing = true;
        Debug.Log("Transcribing...");

        // 1. Get data from Mic
        // We need exactly 480,000 samples (30s * 16000Hz)
        float[] fullBuffer = new float[MaxDuration * SampleRate];
        
        int micPos = Microphone.GetPosition(micDevice);
        
        // Grab the data (safely handling the loop wrap-around is complex, 
        // but this is the simplest "grab recent" method for testing)
        // Note: Ideally you would handle the circular buffer wrap-around here.
        micClip.GetData(fullBuffer, 0);

        // 2. Create the input tensor
        // Because we removed Unity.Barracuda, 'Tensor' now correctly refers to Unity.InferenceEngine.Tensor
        using Tensor<float> audioInput = new Tensor<float>(new TensorShape(1, fullBuffer.Length), fullBuffer);

        // 3. Run Inference 
        // You need to implement the actual Whisper logic here using your workers.
        // Example skeleton:
        // spectroEngine.Schedule(audioInput);
        // var spectroOutput = spectroEngine.PeekOutput() as Tensor<float>;
        // ... pass to encoder ...
        
        yield return null; 

        Debug.Log("Finished Transcription (Check Console)");
        isTranscribing = false;
    }
    
    void SetupEngines()
    {
        // Load the models
        Model modelEncoder = ModelLoader.Load(audioEncoder);
        Model modelDecoder1 = ModelLoader.Load(audioDecoder1);
        Model modelDecoder2 = ModelLoader.Load(audioDecoder2);
        Model modelSpectro = ModelLoader.Load(logMelSpectro);

        // Create the Workers (New Syntax)
        // 'WorkerFactory' is removed. Use 'new Worker()'
        encoderEngine = new Worker(modelEncoder, BackendType.GPUCompute);
        decoderEngine1 = new Worker(modelDecoder1, BackendType.GPUCompute);
        decoderEngine2 = new Worker(modelDecoder2, BackendType.GPUCompute);
        spectroEngine = new Worker(modelSpectro, BackendType.GPUCompute);
    }

    void OnDestroy()
    {
        encoderEngine?.Dispose();
        decoderEngine1?.Dispose();
        decoderEngine2?.Dispose();
        spectroEngine?.Dispose();
    }
}
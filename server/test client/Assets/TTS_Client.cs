using UnityEngine;
using System.Collections;
using System.Text;
using UnityEngine.Networking;
using Firebase.Extensions; 
using Firebase.Auth;

[System.Serializable]
public class AudioRequest
{
    public string text;
    public string languageCode;
    public string voiceName;
}

[System.Serializable]
public class AudioResponse
{
    public string url;
    public bool isCached;
}

[RequireComponent(typeof(AudioSource))]
public class TTS_Client : MonoBehaviour
{
    [SerializeField]
    private string serverEndpoint = "http://localhost:3000/api/get-audio";

    [SerializeField]
    private string textToSpeak = "Hello, this is a secure text-to-speech request from Unity!";
    [SerializeField]
    private string languageCode = "en-US";
    [SerializeField]
    private string voiceName = "en-US-Standard-A";
    [SerializeField]
    private bool play = false;
    private AudioSource audioSource;
    private FirebaseAuth auth;
    private FirebaseUser currentUser;

    private void Awake()
    {
        audioSource = GetComponent<AudioSource>();
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        InitializeFirebase();
    }
    private void InitializeFirebase()
    {
        Firebase.FirebaseApp.CheckAndFixDependenciesAsync().ContinueWithOnMainThread(task => {
            if (task.Result == Firebase.DependencyStatus.Available)
            {
                // Firebase is ready to use
                auth = FirebaseAuth.DefaultInstance;
                SignInAnonymously();
            }
            else
            {
                Debug.LogError("Could not resolve all Firebase dependencies: " + task.Result);
            }
        });
    }
    private void SignInAnonymously()
    {
        auth.SignInAnonymouslyAsync().ContinueWithOnMainThread(task => {
            if (task.IsCanceled || task.IsFaulted)
            {
                Debug.LogError("SignInAnonymouslyAsync encountered an error: " + task.Exception);
                return;
            }

            currentUser = task.Result.User;
            Debug.Log("Firebase Auth Success! User ID: " + currentUser.UserId);

            // Now that we are authenticated, we can safely request audio
            // GenerateAndPlay("Authentication is successful. Generating secure audio.");
        });
    }

    public void GenerateAndPlay(string textToSpeak, string language = "en-US",string voice = null)
    {
        if (currentUser == null)
        {
            Debug.LogError("Wait for Firebase authentication to complete first.");
            return;
        }

        // Fetch the fresh ID Token from Firebase
        currentUser.TokenAsync(true).ContinueWithOnMainThread(task => {
            if (task.IsCanceled || task.IsFaulted)
            {
                Debug.LogError("Failed to get Firebase token: " + task.Exception);
                return;
            }

            string idToken = task.Result;

            // Start the coroutine with the retrieved token
            StartCoroutine(FetchAudioRoutine(textToSpeak, language,voice, idToken));
        });
    }

    private IEnumerator FetchAudioRoutine(string text, string language, string voice, string idToken)
    {
        AudioRequest requestData = new AudioRequest { text = text, languageCode = language ,voiceName = voice};
        string jsonPayload = JsonUtility.ToJson(requestData);

        using (UnityWebRequest postRequest = new UnityWebRequest(serverEndpoint, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);
            postRequest.uploadHandler = new UploadHandlerRaw(bodyRaw);
            postRequest.downloadHandler = new DownloadHandlerBuffer();
            postRequest.SetRequestHeader("Content-Type", "application/json");

            // IMPORTANT: Inject the Firebase ID token into the Authorization header
            postRequest.SetRequestHeader("Authorization", "Bearer " + idToken);

            yield return postRequest.SendWebRequest();

            if (postRequest.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Server Error: " + postRequest.error);
                yield break;
            }

            AudioResponse responseData = JsonUtility.FromJson<AudioResponse>(postRequest.downloadHandler.text);
            StartCoroutine(DownloadAndPlayAudioClip(responseData.url));
        }
    }

    private IEnumerator DownloadAndPlayAudioClip(string fileUrl)
    {
        using (UnityWebRequest audioRequest = UnityWebRequestMultimedia.GetAudioClip(fileUrl, AudioType.MPEG))
        {
            yield return audioRequest.SendWebRequest();

            if (audioRequest.result == UnityWebRequest.Result.Success)
            {
                AudioClip downloadedClip = DownloadHandlerAudioClip.GetContent(audioRequest);
                audioSource.clip = downloadedClip;
                audioSource.Play();
            }
            else
            {
                Debug.LogError("Audio Download Error: " + audioRequest.error);
            }
        }
    }


    // Update is called once per frame
    void Update()
    {
        if (play)
        {
            play = false; // Reset the flag
            GenerateAndPlay(textToSpeak, languageCode, voiceName);
        }
    }
}

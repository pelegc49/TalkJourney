using TalkJourney.BubbleSystem.Audio;
using TalkJourney.BubbleSystem.Localization;
using TalkJourney.GameServices.Auth;
using UnityEngine;

namespace TalkJourney.GameServices
{
    [DefaultExecutionOrder(-2000)]
    [DisallowMultipleComponent]
    public class GlobalGameServicesBootstrap : MonoBehaviour
    {
        [Header("Lifetime")]
        [Tooltip("Keep this GameObject alive across scene loads.")]
        public bool persistAcrossScenes = true;

        [Header("Global Services")]
        [Tooltip("Global localization resolver used across the whole game.")]
        public LocalizationResolver localizationResolver;

        [Tooltip("Global TTS backend client used across the whole game.")]
        public AudioBackendClient audioBackendClient;

        [Tooltip("Global auth token provider used by backend clients.")]
        public FirebaseAuthTokenProvider firebaseAuthTokenProvider;

        [Tooltip("Optional bridge that synchronizes LocalizationResolver and RealtimeWhisper language.")]
        public LanguageSyncBridge languageSyncBridge;

        [Header("Auto Resolve")]
        [Tooltip("Auto-find missing services in the loaded scene.")]
        public bool autoFindMissingServices = true;

        [Header("Settings Persistence")]
        [Tooltip("When enabled, load the saved display language on startup and save changes automatically.")]
        public bool persistDisplayLanguage = true;

        [Tooltip("PlayerPrefs key used to store selected display language.")]
        public string displayLanguagePlayerPrefsKey = "TalkJourney.Settings.DisplayLanguage";

        [Tooltip("When enabled, load the saved native language on startup and save changes automatically.")]
        public bool persistNativeLanguage = true;

        [Tooltip("PlayerPrefs key used to store native language.")]
        public string nativeLanguagePlayerPrefsKey = "TalkJourney.Settings.NativeLanguage";

        [Tooltip("When enabled, load the saved voice name on startup and save changes automatically.")]
        public bool persistVoiceName = true;

        [Tooltip("PlayerPrefs key used to store selected voice name.")]
        public string voiceNamePlayerPrefsKey = "TalkJourney.Settings.VoiceName";

        private static GlobalGameServicesBootstrap _instance;

        public static GlobalGameServicesBootstrap Instance => _instance;

        private void Awake()
        {
            if (_instance != null && _instance != this)
            {
                var keepExistingInstance = _instance.persistAcrossScenes || !persistAcrossScenes;
                if (keepExistingInstance)
                {
                    Debug.Log("GlobalGameServicesBootstrap duplicate detected. Keeping existing instance and removing duplicate bootstrap component.", this);
                    Destroy(this);
                    return;
                }

                Debug.Log("GlobalGameServicesBootstrap duplicate detected. Replacing non-persistent instance bootstrap component with persistent instance.", this);
                Destroy(_instance);
            }

            _instance = this;

            if (persistAcrossScenes)
            {
                DontDestroyOnLoad(gameObject);
            }

            ResolveDependencies();
            WireLanguageBridge();
        }

        private void OnEnable()
        {
            LocalizationResolver.OnDisplayLanguageChanged += OnDisplayLanguageChanged;
            LocalizationResolver.OnLanguagePairChanged += OnLanguagePairChanged;
        }

        private void Start()
        {
            ApplyPersistedLanguagePreferencesIfAvailable();
            ApplyPersistedVoiceIfAvailable();
        }

        private void OnDestroy()
        {
            if (_instance == this)
            {
                _instance = null;
            }
        }

        private void OnDisable()
        {
            LocalizationResolver.OnDisplayLanguageChanged -= OnDisplayLanguageChanged;
            LocalizationResolver.OnLanguagePairChanged -= OnLanguagePairChanged;
        }

        private void ResolveDependencies()
        {
            if (!autoFindMissingServices)
            {
                return;
            }

            if (localizationResolver == null)
            {
                localizationResolver = FindFirstObjectByType<LocalizationResolver>(FindObjectsInactive.Include);
            }

            if (audioBackendClient == null)
            {
                audioBackendClient = FindFirstObjectByType<AudioBackendClient>(FindObjectsInactive.Include);
            }

            if (firebaseAuthTokenProvider == null)
            {
                firebaseAuthTokenProvider = FindFirstObjectByType<FirebaseAuthTokenProvider>(FindObjectsInactive.Include);
            }

            if (languageSyncBridge == null)
            {
                languageSyncBridge = FindFirstObjectByType<LanguageSyncBridge>(FindObjectsInactive.Include);
            }
        }

        private void WireLanguageBridge()
        {
            if (languageSyncBridge == null)
            {
                return;
            }

            if (localizationResolver != null)
            {
                languageSyncBridge.localizationResolver = localizationResolver;
            }

            WireAudioAuthProvider();
        }

        private void WireAudioAuthProvider()
        {
            if (audioBackendClient == null || firebaseAuthTokenProvider == null)
            {
                return;
            }

            if (audioBackendClient.authTokenProviderBehaviour == null)
            {
                audioBackendClient.authTokenProviderBehaviour = firebaseAuthTokenProvider;
            }
        }

        private void ApplyPersistedLanguagePreferencesIfAvailable()
        {
            if (!persistDisplayLanguage || localizationResolver == null || string.IsNullOrWhiteSpace(displayLanguagePlayerPrefsKey))
            {
                return;
            }

            var savedLearningLanguage = localizationResolver.learningLanguage;
            if (PlayerPrefs.HasKey(displayLanguagePlayerPrefsKey))
            {
                var savedLearningValue = PlayerPrefs.GetString(displayLanguagePlayerPrefsKey, DisplayLanguage.English.ToString());
                if (TryParseDisplayLanguage(savedLearningValue, out var parsedLearningLanguage))
                {
                    savedLearningLanguage = parsedLearningLanguage;
                }
            }

            var savedNativeLanguage = localizationResolver.nativeLanguage;
            if (persistNativeLanguage
                && !string.IsNullOrWhiteSpace(nativeLanguagePlayerPrefsKey)
                && PlayerPrefs.HasKey(nativeLanguagePlayerPrefsKey))
            {
                var savedNativeValue = PlayerPrefs.GetString(nativeLanguagePlayerPrefsKey, DisplayLanguage.Hebrew.ToString());
                if (TryParseDisplayLanguage(savedNativeValue, out var parsedNativeLanguage))
                {
                    savedNativeLanguage = parsedNativeLanguage;
                }
            }

            localizationResolver.SetLanguagePair(savedNativeLanguage, savedLearningLanguage);
        }

        private void OnDisplayLanguageChanged(DisplayLanguage language)
        {
            if (!persistDisplayLanguage || string.IsNullOrWhiteSpace(displayLanguagePlayerPrefsKey))
            {
                return;
            }

            PlayerPrefs.SetString(displayLanguagePlayerPrefsKey, language.ToString());
            PlayerPrefs.Save();
        }

        private void OnLanguagePairChanged(DisplayLanguage nativeLanguage, DisplayLanguage _)
        {
            if (!persistNativeLanguage || string.IsNullOrWhiteSpace(nativeLanguagePlayerPrefsKey))
            {
                return;
            }

            PlayerPrefs.SetString(nativeLanguagePlayerPrefsKey, nativeLanguage.ToString());
            PlayerPrefs.Save();
        }

        private static bool TryParseDisplayLanguage(string value, out DisplayLanguage language)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                language = DisplayLanguage.English;
                return false;
            }

            return System.Enum.TryParse(value.Trim(), true, out language);
        }

        private void ApplyPersistedVoiceIfAvailable()
        {
            if (!persistVoiceName || audioBackendClient == null || string.IsNullOrWhiteSpace(voiceNamePlayerPrefsKey))
            {
                return;
            }

            if (!PlayerPrefs.HasKey(voiceNamePlayerPrefsKey))
            {
                return;
            }

            var savedVoiceName = PlayerPrefs.GetString(voiceNamePlayerPrefsKey, string.Empty);
            if (!string.IsNullOrWhiteSpace(savedVoiceName))
            {
                audioBackendClient.voiceName = savedVoiceName;
            }
        }

        public void SetVoiceNamePreference(string voiceName)
        {
            if (!persistVoiceName || string.IsNullOrWhiteSpace(voiceNamePlayerPrefsKey))
            {
                return;
            }

            if (audioBackendClient != null)
            {
                audioBackendClient.voiceName = voiceName;
            }

            PlayerPrefs.SetString(voiceNamePlayerPrefsKey, voiceName);
            PlayerPrefs.Save();
        }

        public string GetVoiceNamePreference()
        {
            if (!persistVoiceName || string.IsNullOrWhiteSpace(voiceNamePlayerPrefsKey))
            {
                return null;
            }

            return PlayerPrefs.HasKey(voiceNamePlayerPrefsKey) 
                ? PlayerPrefs.GetString(voiceNamePlayerPrefsKey) 
                : null;
        }
    }
}

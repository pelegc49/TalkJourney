using TalkJourney.BubbleSystem.Audio;
using TalkJourney.BubbleSystem.Localization;
using TMPro;
using UnityEngine;

namespace TalkJourney.GameServices.Settings
{
    /// <summary>
    /// Game-wide voice settings UI. Allows selecting TTS voice variants per language.
    /// Persists voice preference independently from BubbleSystem lifecycle.
    /// </summary>
    public class GlobalVoiceSettingsUI : MonoBehaviour
    {
        [SerializeField]
        private TMP_Dropdown voiceDropdown;

        [SerializeField]
        private AudioBackendClient audioBackendClient;

        [SerializeField]
        private LocalizationResolver localizationResolver;

        private bool _isUpdatingDropdown;

        private void Awake()
        {
            if (voiceDropdown == null)
            {
                Debug.LogError("GlobalVoiceSettingsUI: No Dropdown assigned. Please assign a Dropdown component.", this);
                return;
            }

            ResolveAudioBackendClient();
            ResolveLocalizationResolver();
            InitializeVoiceDropdown();
        }

        private void OnEnable()
        {
            if (voiceDropdown != null)
            {
                voiceDropdown.onValueChanged.AddListener(OnVoiceSelected);
            }

            LocalizationResolver.OnLanguagePairChanged += OnLanguagePairChanged;
            LocalizationResolver.OnDisplayLanguageChanged += OnDisplayLanguageChanged;

            RefreshVoiceOptionsForLearningLanguage();
            SyncDropdownSelectionToCurrentVoice();
        }

        private void OnDisable()
        {
            if (voiceDropdown != null)
            {
                voiceDropdown.onValueChanged.RemoveListener(OnVoiceSelected);
            }

            LocalizationResolver.OnLanguagePairChanged -= OnLanguagePairChanged;
            LocalizationResolver.OnDisplayLanguageChanged -= OnDisplayLanguageChanged;
        }

        private void ResolveAudioBackendClient()
        {
            if (audioBackendClient != null)
            {
                return;
            }

            var globalServices = GlobalGameServicesBootstrap.Instance;
            if (globalServices != null && globalServices.audioBackendClient != null)
            {
                audioBackendClient = globalServices.audioBackendClient;
                return;
            }

            audioBackendClient = FindFirstObjectByType<AudioBackendClient>(FindObjectsInactive.Include);
        }

        private void ResolveLocalizationResolver()
        {
            if (localizationResolver != null)
            {
                return;
            }

            var globalServices = GlobalGameServicesBootstrap.Instance;
            if (globalServices != null && globalServices.localizationResolver != null)
            {
                localizationResolver = globalServices.localizationResolver;
                return;
            }

            localizationResolver = FindFirstObjectByType<LocalizationResolver>(FindObjectsInactive.Include);
        }

        private void InitializeVoiceDropdown()
        {
            if (voiceDropdown == null)
            {
                return;
            }

            RefreshVoiceOptionsForLearningLanguage();
        }

        private void OnVoiceSelected(int index)
        {
            if (_isUpdatingDropdown)
            {
                return;
            }

            if (voiceDropdown == null || index < 0 || index >= voiceDropdown.options.Count)
            {
                return;
            }

            var selectedVoice = voiceDropdown.options[index].text;

            if (audioBackendClient != null)
            {
                audioBackendClient.voiceName = selectedVoice;
            }

            var globalServices = GlobalGameServicesBootstrap.Instance;
            if (globalServices != null)
            {
                globalServices.SetVoiceNamePreference(selectedVoice);
            }
        }

        private void OnLanguagePairChanged(DisplayLanguage _, DisplayLanguage learningLanguage)
        {
            RefreshVoiceOptionsForLanguage(learningLanguage);
        }

        private void OnDisplayLanguageChanged(DisplayLanguage learningLanguage)
        {
            RefreshVoiceOptionsForLanguage(learningLanguage);
        }

        private void RefreshVoiceOptionsForLearningLanguage()
        {
            var learningLanguage = localizationResolver != null ? localizationResolver.learningLanguage : DisplayLanguage.English;
            RefreshVoiceOptionsForLanguage(learningLanguage);
        }

        private void RefreshVoiceOptionsForLanguage(DisplayLanguage learningLanguage)
        {
            if (voiceDropdown == null)
            {
                return;
            }

            _isUpdatingDropdown = true;
            voiceDropdown.options.Clear();

            var voiceOptions = GetVoiceOptionsForLanguage(learningLanguage);
            for (int i = 0; i < voiceOptions.Length; i++)
            {
                voiceDropdown.options.Add(new TMP_Dropdown.OptionData(voiceOptions[i]));
            }

            EnsureCurrentVoiceMatchesAvailableOptions(voiceOptions);
            SyncDropdownSelectionToCurrentVoice();
            _isUpdatingDropdown = false;
        }

        private void EnsureCurrentVoiceMatchesAvailableOptions(string[] availableOptions)
        {
            if (audioBackendClient == null || availableOptions == null || availableOptions.Length == 0)
            {
                return;
            }

            var currentVoice = audioBackendClient.voiceName;
            for (int i = 0; i < availableOptions.Length; i++)
            {
                if (string.Equals(currentVoice, availableOptions[i], System.StringComparison.OrdinalIgnoreCase))
                {
                    return;
                }
            }

            var defaultVoice = availableOptions[0];
            audioBackendClient.voiceName = defaultVoice;

            var globalServices = GlobalGameServicesBootstrap.Instance;
            if (globalServices != null)
            {
                globalServices.SetVoiceNamePreference(defaultVoice);
            }
        }

        private static string[] GetVoiceOptionsForLanguage(DisplayLanguage language)
        {
            switch (language)
            {
                case DisplayLanguage.Hebrew:
                    return new[]
                    {
                        "he-IL-Standard-A",
                        "he-IL-Standard-B"
                    };
                case DisplayLanguage.Russian:
                    return new[]
                    {
                        "ru-RU-Standard-A",
                        "ru-RU-Standard-B"
                    };
                case DisplayLanguage.English:
                default:
                    return new[]
                    {
                        "en-US-Standard-A",
                        "en-US-Standard-B",
                        "en-US-Standard-C",
                        "en-US-Standard-D"
                    };
            }
        }

        private void SyncDropdownSelectionToCurrentVoice()
        {
            if (voiceDropdown == null || audioBackendClient == null)
            {
                return;
            }

            var currentVoice = audioBackendClient.voiceName;
            if (string.IsNullOrWhiteSpace(currentVoice))
            {
                return;
            }

            var selectedIndex = 0;
            for (int i = 0; i < voiceDropdown.options.Count; i++)
            {
                var optionText = voiceDropdown.options[i].text;
                if (string.Equals(optionText, currentVoice, System.StringComparison.OrdinalIgnoreCase))
                {
                    selectedIndex = i;
                    break;
                }
            }

            _isUpdatingDropdown = true;
            voiceDropdown.SetValueWithoutNotify(selectedIndex);
            _isUpdatingDropdown = false;
        }

        private void OnDestroy()
        {
            if (voiceDropdown != null)
            {
                voiceDropdown.onValueChanged.RemoveListener(OnVoiceSelected);
            }
        }
    }
}

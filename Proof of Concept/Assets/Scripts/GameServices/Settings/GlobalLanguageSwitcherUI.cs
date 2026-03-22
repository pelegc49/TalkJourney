using System.Collections.Generic;
using TalkJourney.BubbleSystem.Localization;
using TMPro;
using UnityEngine;
using UnityEngine.Localization;
using UnityEngine.Localization.Settings;
using UnityEngine.Serialization;

namespace TalkJourney.GameServices.Settings
{
    /// <summary>
    /// Game-wide language settings UI. This component is service-oriented and not tied to BubbleSystem lifecycle.
    /// </summary>
    public class GlobalLanguageSwitcherUI : MonoBehaviour
    {
        [SerializeField]
        private TMP_Dropdown nativeLanguageDropdown;

        [SerializeField]
        [FormerlySerializedAs("languageDropdown")]
        private TMP_Dropdown learningLanguageDropdown;

        [SerializeField]
        private LocalizationResolver localizationResolver;

        private readonly List<Locale> _nativeDropdownLocales = new List<Locale>(8);
        private readonly List<Locale> _learningDropdownLocales = new List<Locale>(8);
        private bool _isUpdatingNativeDropdown;
        private bool _isUpdatingLearningDropdown;

        private void Awake()
        {
            if (nativeLanguageDropdown == null || learningLanguageDropdown == null)
            {
                Debug.LogError("GlobalLanguageSwitcherUI: Assign both nativeLanguageDropdown and learningLanguageDropdown.", this);
                return;
            }

            ResolveLocalizationResolver();
            InitializeDropdowns();
        }

        private void OnEnable()
        {
            if (nativeLanguageDropdown != null)
            {
                nativeLanguageDropdown.onValueChanged.AddListener(OnNativeLanguageSelected);
            }

            if (learningLanguageDropdown != null)
            {
                learningLanguageDropdown.onValueChanged.AddListener(OnLearningLanguageSelected);
            }

            LocalizationResolver.OnLanguagePairChanged += OnLanguagePairChanged;
            LocalizationResolver.OnDisplayLanguageChanged += OnLearningLanguageChanged;
            SyncDropdownSelectionsToCurrentPair();
        }

        private void OnDisable()
        {
            if (nativeLanguageDropdown != null)
            {
                nativeLanguageDropdown.onValueChanged.RemoveListener(OnNativeLanguageSelected);
            }

            if (learningLanguageDropdown != null)
            {
                learningLanguageDropdown.onValueChanged.RemoveListener(OnLearningLanguageSelected);
            }

            LocalizationResolver.OnLanguagePairChanged -= OnLanguagePairChanged;
            LocalizationResolver.OnDisplayLanguageChanged -= OnLearningLanguageChanged;
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

        private void InitializeDropdowns()
        {
            if (nativeLanguageDropdown == null || learningLanguageDropdown == null)
            {
                return;
            }

            _isUpdatingNativeDropdown = true;
            _isUpdatingLearningDropdown = true;
            nativeLanguageDropdown.options.Clear();
            learningLanguageDropdown.options.Clear();
            _nativeDropdownLocales.Clear();
            _learningDropdownLocales.Clear();

            var availableLocales = LocalizationSettings.AvailableLocales;
            if (availableLocales?.Locales == null || availableLocales.Locales.Count == 0)
            {
                Debug.LogWarning("GlobalLanguageSwitcherUI: No locales are available.");
                _isUpdatingNativeDropdown = false;
                _isUpdatingLearningDropdown = false;
                return;
            }

            for (int i = 0; i < availableLocales.Locales.Count; i++)
            {
                var locale = availableLocales.Locales[i];
                if (locale == null || locale.name.Contains('-'))
                {
                    continue;
                }

                string displayName = !string.IsNullOrEmpty(locale.name) ? locale.name : locale.Identifier.Code;

                nativeLanguageDropdown.options.Add(new TMP_Dropdown.OptionData(displayName));
                learningLanguageDropdown.options.Add(new TMP_Dropdown.OptionData(displayName));
                _nativeDropdownLocales.Add(locale);
                _learningDropdownLocales.Add(locale);
            }

            SyncDropdownSelectionsToCurrentPair();
            _isUpdatingNativeDropdown = false;
            _isUpdatingLearningDropdown = false;
        }

        private void OnNativeLanguageSelected(int index)
        {
            if (_isUpdatingNativeDropdown)
            {
                return;
            }

            if (index < 0 || index >= _nativeDropdownLocales.Count)
            {
                return;
            }

            if (!TryLocaleToDisplayLanguage(_nativeDropdownLocales[index], out var nativeLanguage))
            {
                return;
            }

            if (localizationResolver != null)
            {
                localizationResolver.SetNativeLanguage(nativeLanguage);
            }
        }

        private void OnLearningLanguageSelected(int index)
        {
            if (_isUpdatingLearningDropdown)
            {
                return;
            }

            if (index < 0 || index >= _learningDropdownLocales.Count)
            {
                return;
            }

            var selectedLocale = _learningDropdownLocales[index];
            if (!TryLocaleToDisplayLanguage(selectedLocale, out var learningLanguage))
            {
                return;
            }

            if (localizationResolver != null)
            {
                localizationResolver.SetLearningLanguage(learningLanguage);
                return;
            }

            LocalizationSettings.SelectedLocale = selectedLocale;
            Debug.Log($"Learning language changed to: {selectedLocale.name}");
        }

        private void OnLanguagePairChanged(DisplayLanguage _, DisplayLanguage __)
        {
            SyncDropdownSelectionsToCurrentPair();
        }

        private void OnLearningLanguageChanged(DisplayLanguage _)
        {
            SyncDropdownSelectionsToCurrentPair();
        }

        private void SyncDropdownSelectionsToCurrentPair()
        {
            if (nativeLanguageDropdown == null || learningLanguageDropdown == null)
            {
                return;
            }

            if (_nativeDropdownLocales.Count == 0 || _learningDropdownLocales.Count == 0)
            {
                return;
            }

            var nativeLanguage = localizationResolver != null ? localizationResolver.nativeLanguage : DisplayLanguage.Hebrew;
            var learningLanguage = localizationResolver != null ? localizationResolver.learningLanguage : DisplayLanguage.English;

            var nativeCode = DisplayLanguageToLocaleCode(nativeLanguage);
            var learningCode = DisplayLanguageToLocaleCode(learningLanguage);

            var nativeIndex = FindLocaleIndex(_nativeDropdownLocales, nativeCode);
            var learningIndex = FindLocaleIndex(_learningDropdownLocales, learningCode);

            _isUpdatingNativeDropdown = true;
            nativeLanguageDropdown.SetValueWithoutNotify(nativeIndex);
            _isUpdatingNativeDropdown = false;

            _isUpdatingLearningDropdown = true;
            learningLanguageDropdown.SetValueWithoutNotify(learningIndex);
            _isUpdatingLearningDropdown = false;
        }

        private static int FindLocaleIndex(List<Locale> locales, string localeCode)
        {
            var selectedIndex = 0;
            for (int i = 0; i < locales.Count; i++)
            {
                var locale = locales[i];
                if (locale != null && string.Equals(locale.Identifier.Code, localeCode, System.StringComparison.OrdinalIgnoreCase))
                {
                    selectedIndex = i;
                    break;
                }
            }

            return selectedIndex;
        }

        private static string DisplayLanguageToLocaleCode(DisplayLanguage language)
        {
            switch (language)
            {
                case DisplayLanguage.English:
                    return "en";
                case DisplayLanguage.Hebrew:
                    return "he";
                case DisplayLanguage.Russian:
                    return "ru";
                default:
                    return "en";
            }
        }

        private static bool TryLocaleToDisplayLanguage(Locale locale, out DisplayLanguage language)
        {
            language = DisplayLanguage.English;
            if (locale == null || string.IsNullOrWhiteSpace(locale.Identifier.Code))
            {
                return false;
            }

            switch (locale.Identifier.Code.Trim().ToLowerInvariant())
            {
                case "en":
                    language = DisplayLanguage.English;
                    return true;
                case "he":
                    language = DisplayLanguage.Hebrew;
                    return true;
                case "ru":
                    language = DisplayLanguage.Russian;
                    return true;
                default:
                    return false;
            }
        }

        private void OnDestroy()
        {
            if (nativeLanguageDropdown != null)
            {
                nativeLanguageDropdown.onValueChanged.RemoveListener(OnNativeLanguageSelected);
            }

            if (learningLanguageDropdown != null)
            {
                learningLanguageDropdown.onValueChanged.RemoveListener(OnLearningLanguageSelected);
            }
        }
    }
}

using System.Collections.Generic;
using System;
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

        private static readonly DisplayLanguage[] DropdownLanguageOrder =
        {
            DisplayLanguage.English,
            DisplayLanguage.Hebrew,
            DisplayLanguage.Russian
        };

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

            var bestLocaleByLanguage = new Dictionary<DisplayLanguage, Locale>(3);
            var bestScoreByLanguage = new Dictionary<DisplayLanguage, int>(3);

            for (int i = 0; i < availableLocales.Locales.Count; i++)
            {
                var locale = availableLocales.Locales[i];
                if (!TryGetDisplayLanguage(locale, out var language))
                {
                    continue;
                }

                var score = GetDisplayLocalePriority(locale.Identifier.Code);
                if (!bestScoreByLanguage.TryGetValue(language, out var previousScore) || score > previousScore)
                {
                    bestScoreByLanguage[language] = score;
                    bestLocaleByLanguage[language] = locale;
                }
            }

            for (int i = 0; i < DropdownLanguageOrder.Length; i++)
            {
                var language = DropdownLanguageOrder[i];
                if (!bestLocaleByLanguage.TryGetValue(language, out var locale))
                {
                    continue;
                }

                var displayName = DisplayLanguageToLabel(language);
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
            if (!TryLocaleCodeToDisplayLanguage(localeCode, out var targetLanguage))
            {
                targetLanguage = DisplayLanguage.English;
            }

            var selectedIndex = 0;
            for (int i = 0; i < locales.Count; i++)
            {
                var locale = locales[i];
                if (locale == null)
                {
                    continue;
                }

                if (TryGetDisplayLanguage(locale, out var localeLanguage) && localeLanguage == targetLanguage)
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
            if (locale == null)
            {
                language = DisplayLanguage.English;
                return false;
            }

            return TryGetDisplayLanguage(locale, out language);
        }

        private static bool TryGetDisplayLanguage(Locale locale, out DisplayLanguage language)
        {
            language = DisplayLanguage.English;
            if (locale == null || string.IsNullOrWhiteSpace(locale.Identifier.Code))
            {
                return false;
            }

            return TryLocaleCodeToDisplayLanguage(locale.Identifier.Code, out language);
        }

        private static bool TryLocaleCodeToDisplayLanguage(string localeCode, out DisplayLanguage language)
        {
            language = DisplayLanguage.English;
            if (string.IsNullOrWhiteSpace(localeCode))
            {
                return false;
            }

            var normalized = localeCode.Trim();
            var parts = normalized.Split('-');
            if (parts.Length == 0)
            {
                return false;
            }

            var baseCode = parts[0].ToLowerInvariant();

            // Reject transliterator pair locales (for example en-he, he-ru) from UI language dropdowns.
            if (parts.Length > 1)
            {
                var secondPart = parts[1];
                if (secondPart.Length == 2 && secondPart.Equals(secondPart.ToLowerInvariant(), StringComparison.Ordinal))
                {
                    return false;
                }
            }

            switch (baseCode)
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

        private static int GetDisplayLocalePriority(string localeCode)
        {
            if (string.IsNullOrWhiteSpace(localeCode))
            {
                return 0;
            }

            var normalized = localeCode.Trim();
            if (normalized.Equals("en-US", StringComparison.OrdinalIgnoreCase)
                || normalized.Equals("he-IL", StringComparison.OrdinalIgnoreCase)
                || normalized.Equals("ru-RU", StringComparison.OrdinalIgnoreCase))
            {
                return 3;
            }

            if (normalized.Equals("en", StringComparison.OrdinalIgnoreCase)
                || normalized.Equals("he", StringComparison.OrdinalIgnoreCase)
                || normalized.Equals("ru", StringComparison.OrdinalIgnoreCase))
            {
                return 2;
            }

            return 1;
        }

        private static string DisplayLanguageToLabel(DisplayLanguage language)
        {
            switch (language)
            {
                case DisplayLanguage.English:
                    return "English";
                case DisplayLanguage.Hebrew:
                    return "Hebrew";
                case DisplayLanguage.Russian:
                    return "Russian";
                default:
                    return "English";
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

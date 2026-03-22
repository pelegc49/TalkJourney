using System;
using UnityEngine;
using UnityEngine.Localization;
using UnityEngine.Localization.Settings;
using UnityEngine.Localization.Tables;

namespace TalkJourney.BubbleSystem.Localization
{
    /// Available languages that can be selected from the Inspector.
    /// Add more languages here as needed.
    public enum DisplayLanguage
    {
        English,
        Hebrew,
        Russian
    }

    /// <summary>
    /// Transliterator modes that convert phonetic pronunciation from one language to another script.
    /// Format: Source Language - Target Script
    /// For example: en-he means English source with Hebrew script target output
    /// </summary>
    public enum TransliteratorMode
    {
        EnglishToHebrew,  // en-he
        EnglishToRussian, // en-ru
        HebrewToEnglish,  // he-en
        HebrewToRussian,  // he-ru
        RussianToEnglish, // ru-en
        RussianToHebrew   // ru-he
    }

    [DisallowMultipleComponent]
    public class LocalizationResolver : MonoBehaviour, ILocalizationService
    {
        /// <summary>
        /// Event that fires whenever the display language changes.
        /// Subscribe to this to refresh UI text when the language changes mid-game.
        /// </summary>
        public static event System.Action OnLanguageChanged;

        /// <summary>
        /// Event with payload for systems that need the selected display language value.
        /// </summary>
        public static event System.Action<DisplayLanguage> OnDisplayLanguageChanged;

        /// <summary>
        /// Event that fires whenever the transliterator mode changes.
        /// Subscribe to this to refresh transliteration text when the mode changes mid-game.
        /// </summary>
        public static event System.Action OnTransliteratorChanged;

        /// <summary>
        /// Event that fires whenever native/learning language pair changes.
        /// Payload order: native language, learning language.
        /// </summary>
        public static event System.Action<DisplayLanguage, DisplayLanguage> OnLanguagePairChanged;

        [Header("Unity Localization")]
        [Tooltip("Default String Table Collection used when the key does not include an explicit table prefix.")]
        public string defaultStringTableCollection = "FirstScene";

        [Tooltip("When true, waits for LocalizationSettings initialization before first resolve.")]
        public bool waitForLocalizationInitialization = true;

        [Header("Fallback Behavior")]
        [Tooltip("If true, unresolved keys return the key itself. If false, unresolved keys return an empty string.")]
        public bool returnKeyWhenMissing = true;

        [Header("Display Language")]
        [Tooltip("Legacy display language. This now follows learningLanguage.")]
        public DisplayLanguage selectedLanguage = DisplayLanguage.English;

        private DisplayLanguage previousLanguage = DisplayLanguage.English;

        [Header("Language Pair")]
        [Tooltip("User native language (target script for transliteration).")]
        public DisplayLanguage nativeLanguage = DisplayLanguage.Hebrew;

        [Tooltip("Language being learned. This drives display language and speech recognition language.")]
        public DisplayLanguage learningLanguage = DisplayLanguage.English;

        private DisplayLanguage previousNativeLanguage = DisplayLanguage.Hebrew;
        private DisplayLanguage previousLearningLanguage = DisplayLanguage.English;

        [Header("Transliterator")]
        [Tooltip("Select the transliterator mode (source-target script conversion). Changing this in the Inspector will immediately update transliterations.")]
        public TransliteratorMode selectedTransliterator = TransliteratorMode.EnglishToHebrew;

        private TransliteratorMode previousTransliterator = TransliteratorMode.EnglishToHebrew;

        private void Awake()
        {
            EnsureLocalizationInitialized();
            selectedLanguage = learningLanguage;
            previousLanguage = selectedLanguage;
            previousLearningLanguage = learningLanguage;
            previousNativeLanguage = nativeLanguage;
            ApplyLanguagePair();
        }

        private void OnValidate()
        {
            // Keep legacy selectedLanguage field synchronized as learning language.
            if (selectedLanguage != learningLanguage)
            {
                learningLanguage = selectedLanguage;
            }

            var pairChanged = nativeLanguage != previousNativeLanguage || learningLanguage != previousLearningLanguage;
            if (pairChanged)
            {
                previousNativeLanguage = nativeLanguage;
                previousLearningLanguage = learningLanguage;
                selectedLanguage = learningLanguage;
                previousLanguage = selectedLanguage;

                if (TryMapLanguagePairToTransliterator(learningLanguage, nativeLanguage, out var mappedMode))
                {
                    selectedTransliterator = mappedMode;
                    previousTransliterator = selectedTransliterator;
                }

                #if UNITY_EDITOR
                if (!Application.isPlaying)
                {
                    // In editor, preview locale from learning language.
                    string localeCode = EnumToLocaleCode(learningLanguage);
                    var availableLocales = LocalizationSettings.AvailableLocales;
                    if (availableLocales?.Locales != null)
                    {
                        foreach (var locale in availableLocales.Locales)
                        {
                            if (locale != null && locale.Identifier.Code == localeCode)
                            {
                                LocalizationSettings.SelectedLocale = locale;
                                break;
                            }
                        }
                    }
                }
                else
                {
                    SetLanguagePair(nativeLanguage, learningLanguage);
                }
                #endif
            }

            if (selectedTransliterator != previousTransliterator)
            {
                previousTransliterator = selectedTransliterator;
                if (Application.isPlaying)
                {
                    OnTransliteratorChanged?.Invoke();
                }
            }
        }

        private void ApplyLanguagePair()
        {
            SetLanguagePair(nativeLanguage, learningLanguage);
        }

        /// <summary>
        /// Sets display language from enum and applies the corresponding locale.
        /// </summary>
        public bool SetDisplayLanguage(DisplayLanguage language)
        {
            return SetLearningLanguage(language);
        }

        public bool SetNativeLanguage(DisplayLanguage language)
        {
            var adjustedLearning = learningLanguage;
            if (language == adjustedLearning)
            {
                adjustedLearning = GetAlternativeLanguage(language);
            }

            return SetLanguagePair(language, adjustedLearning);
        }

        public bool SetLearningLanguage(DisplayLanguage language)
        {
            var adjustedNative = nativeLanguage;
            if (language == adjustedNative)
            {
                adjustedNative = GetAlternativeLanguage(language);
            }

            return SetLanguagePair(adjustedNative, language);
        }

        public bool SetLanguagePair(DisplayLanguage native, DisplayLanguage learning)
        {
            if (native == learning)
            {
                native = GetAlternativeLanguage(learning);
            }

            var pairChanged = nativeLanguage != native || learningLanguage != learning;

            nativeLanguage = native;
            learningLanguage = learning;
            previousNativeLanguage = native;
            previousLearningLanguage = learning;

            selectedLanguage = learning;
            previousLanguage = selectedLanguage;

            if (TryMapLanguagePairToTransliterator(learningLanguage, nativeLanguage, out var mappedMode)
                && selectedTransliterator != mappedMode)
            {
                selectedTransliterator = mappedMode;
                previousTransliterator = selectedTransliterator;
                OnTransliteratorChanged?.Invoke();
            }

            var isDisplaySet = SetDisplayLanguage(EnumToLocaleCode(learningLanguage));

            if (pairChanged)
            {
                OnLanguagePairChanged?.Invoke(nativeLanguage, learningLanguage);
            }

            return isDisplaySet;
        }

        private static DisplayLanguage GetAlternativeLanguage(DisplayLanguage language)
        {
            switch (language)
            {
                case DisplayLanguage.English:
                    return DisplayLanguage.Hebrew;
                case DisplayLanguage.Hebrew:
                    return DisplayLanguage.English;
                case DisplayLanguage.Russian:
                    return DisplayLanguage.English;
                default:
                    return DisplayLanguage.English;
            }
        }

        private string EnumToLocaleCode(DisplayLanguage language)
        {
            return language switch
            {
                DisplayLanguage.English => "en",
                DisplayLanguage.Hebrew => "he",
                DisplayLanguage.Russian => "ru",
                _ => "en"
            };
        }

        /// <summary>
        /// Converts a TransliteratorMode enum to its locale code string (e.g., "en-he", "he-ru").
        /// </summary>
        private string EnumToTransliteratorCode(TransliteratorMode mode)
        {
            return mode switch
            {
                TransliteratorMode.EnglishToHebrew => "en-he",
                TransliteratorMode.EnglishToRussian => "en-ru",
                TransliteratorMode.HebrewToEnglish => "he-en",
                TransliteratorMode.HebrewToRussian => "he-ru",
                TransliteratorMode.RussianToEnglish => "ru-en",
                TransliteratorMode.RussianToHebrew => "ru-he",
                _ => "en-he"
            };
        }

        /// <summary>
        /// Returns only the valid transliterator modes for a given source language.
        /// If the source language is English, returns modes that start with "en-" (target to Hebrew or Russian).
        /// </summary>
        private System.Collections.Generic.List<TransliteratorMode> GetValidTransliteratorModesForLanguage(DisplayLanguage language)
        {
            var validModes = new System.Collections.Generic.List<TransliteratorMode>();

            return language switch
            {
                DisplayLanguage.English => new System.Collections.Generic.List<TransliteratorMode>
                {
                    TransliteratorMode.EnglishToHebrew,
                    TransliteratorMode.EnglishToRussian
                },
                DisplayLanguage.Hebrew => new System.Collections.Generic.List<TransliteratorMode>
                {
                    TransliteratorMode.HebrewToEnglish,
                    TransliteratorMode.HebrewToRussian
                },
                DisplayLanguage.Russian => new System.Collections.Generic.List<TransliteratorMode>
                {
                    TransliteratorMode.RussianToEnglish,
                    TransliteratorMode.RussianToHebrew
                },
                _ => new System.Collections.Generic.List<TransliteratorMode> { TransliteratorMode.EnglishToHebrew }
            };
        }

        /// <summary>
        /// Gets a display name for a transliterator mode (e.g., "To Hebrew", "To Russian").
        /// </summary>
        private string GetTransliteratorModeName(TransliteratorMode mode)
        {
            return mode switch
            {
                TransliteratorMode.EnglishToHebrew => "To Hebrew",
                TransliteratorMode.EnglishToRussian => "To Russian",
                TransliteratorMode.HebrewToEnglish => "To English",
                TransliteratorMode.HebrewToRussian => "To Russian",
                TransliteratorMode.RussianToEnglish => "To English",
                TransliteratorMode.RussianToHebrew => "To Hebrew",
                _ => "Unknown"
            };
        }

        /// <summary>
        /// Checks if a transliterator mode is valid for the currently selected display language.
        /// </summary>
        private bool IsTransliteratorValidForLanguage(TransliteratorMode mode, DisplayLanguage language)
        {
            return GetValidTransliteratorModesForLanguage(language).Contains(mode);
        }

        /// <summary>
        /// Gets the current transliterator locale code based on the selected mode.
        /// </summary>
        public string GetCurrentTransliteratorCode()
        {
            if (TryMapLanguagePairToTransliterator(learningLanguage, nativeLanguage, out var mappedMode))
            {
                return EnumToTransliteratorCode(mappedMode);
            }

            return EnumToTransliteratorCode(selectedTransliterator);
        }

        public string Resolve(string key)
        {
            if (TryResolve(key, out var localizedValue))
            {
                return localizedValue;
            }

            return returnKeyWhenMissing ? key : string.Empty;
        }

        public bool TryResolve(string key, out string localizedValue)
        {
            return TryResolveInternal(key, null, out localizedValue);
        }

        public string ResolveForLocaleCode(string key, string localeCode)
        {
            if (TryResolveForLocaleCode(key, localeCode, out var localizedValue))
            {
                return localizedValue;
            }

            return returnKeyWhenMissing ? key : string.Empty;
        }

        public bool TryResolveForLocaleCode(string key, string localeCode, out string localizedValue)
        {
            return TryResolveInternal(key, localeCode, out localizedValue);
        }

        /// Sets the display language by locale code (e.g., "en", "he", "ru").
        /// This changes the language used throughout the bubble system without using the default UI switcher.
        /// @param localeCode : The locale code to set (e.g., "en" for English, "he" for Hebrew, "ru" for Russian).
        /// @return : True if the locale was successfully set, false if the locale code was not found.
        public bool SetDisplayLanguage(string localeCode)
        {
            if (string.IsNullOrWhiteSpace(localeCode))
            {
                Debug.LogWarning("Locale code cannot be null or empty.");
                return false;
            }

            EnsureLocalizationInitialized();

            var availableLocales = LocalizationSettings.AvailableLocales;
            if (availableLocales?.Locales == null || availableLocales.Locales.Count == 0)
            {
                Debug.LogWarning("No locales are available in LocalizationSettings.");
                return false;
            }

            for (int i = 0; i < availableLocales.Locales.Count; i++)
            {
                var locale = availableLocales.Locales[i];
                if (locale != null && string.Equals(locale.Identifier.Code, localeCode.Trim(), StringComparison.OrdinalIgnoreCase))
                {
                    LocalizationSettings.SelectedLocale = locale;
                    if (TryLocaleCodeToDisplayLanguage(locale.Identifier.Code, out var displayLanguage))
                    {
                        selectedLanguage = displayLanguage;
                        previousLanguage = displayLanguage;
                        learningLanguage = displayLanguage;
                        previousLearningLanguage = displayLanguage;

                        if (TryMapLanguagePairToTransliterator(learningLanguage, nativeLanguage, out var mappedMode)
                            && selectedTransliterator != mappedMode)
                        {
                            selectedTransliterator = mappedMode;
                            previousTransliterator = selectedTransliterator;
                            OnTransliteratorChanged?.Invoke();
                        }
                    }
                    Debug.Log($"Display language changed to: {localeCode} ({locale.name})");
                    
                    // Fire event to notify all listeners (bubbles) to refresh their text
                    OnLanguageChanged?.Invoke();
                    OnDisplayLanguageChanged?.Invoke(selectedLanguage);
                    OnLanguagePairChanged?.Invoke(nativeLanguage, learningLanguage);
                    
                    return true;
                }
            }

            Debug.LogWarning($"Locale code '{localeCode}' not found in available locales.");
            return false;
        }

        private bool TryResolveInternal(string key, string localeCode, out string localizedValue)
        {
            localizedValue = string.Empty;

            if (string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            EnsureLocalizationInitialized();

            if (!TryParseTableAndEntry(key, out var tableName, out var entryKey))
            {
                return false;
            }

            if (string.IsNullOrWhiteSpace(tableName) || string.IsNullOrWhiteSpace(entryKey))
            {
                return false;
            }

            var locale = ResolveLocale(localeCode);
            var table = locale == null
                ? LocalizationSettings.StringDatabase.GetTable(tableName)
                : LocalizationSettings.StringDatabase.GetTable(tableName, locale);

            if (table == null)
            {
                return false;
            }

            var stringTable = table as StringTable;
            if (stringTable == null)
            {
                return false;
            }

            var tableEntry = stringTable.GetEntry(entryKey);
            if (tableEntry == null)
            {
                return false;
            }

            localizedValue = tableEntry.LocalizedValue ?? string.Empty;
            return true;
        }

        private Locale ResolveLocale(string localeCode)
        {
            if (string.IsNullOrWhiteSpace(localeCode))
            {
                return LocalizationSettings.SelectedLocale;
            }

            var availableLocales = LocalizationSettings.AvailableLocales;
            if (availableLocales == null)
            {
                return null;
            }

            for (int i = 0; i < availableLocales.Locales.Count; i++)
            {
                var locale = availableLocales.Locales[i];
                if (locale == null)
                {
                    continue;
                }

                if (string.Equals(locale.Identifier.Code, localeCode.Trim(), StringComparison.OrdinalIgnoreCase))
                {
                    return locale;
                }
            }

            return null;
        }

        private void EnsureLocalizationInitialized()
        {
            if (!waitForLocalizationInitialization)
            {
                return;
            }

            var init = LocalizationSettings.InitializationOperation;
            if (!init.IsDone)
            {
                init.WaitForCompletion();
            }
        }

        private bool TryParseTableAndEntry(string rawKey, out string tableName, out string entryKey)
        {
            tableName = defaultStringTableCollection;
            entryKey = rawKey.Trim();

            var slashIndex = entryKey.IndexOf('/');
            if (slashIndex > 0 && slashIndex < entryKey.Length - 1)
            {
                tableName = entryKey.Substring(0, slashIndex).Trim();
                entryKey = entryKey.Substring(slashIndex + 1).Trim();
                return true;
            }

            var colonIndex = entryKey.IndexOf(':');
            if (colonIndex > 0 && colonIndex < entryKey.Length - 1)
            {
                tableName = entryKey.Substring(0, colonIndex).Trim();
                entryKey = entryKey.Substring(colonIndex + 1).Trim();
                return true;
            }

            if (string.IsNullOrWhiteSpace(entryKey))
            {
                return false;
            }

            return true;
        }

        private bool TryLocaleCodeToDisplayLanguage(string localeCode, out DisplayLanguage language)
        {
            language = DisplayLanguage.English;

            if (string.IsNullOrWhiteSpace(localeCode))
            {
                return false;
            }

            switch (localeCode.Trim().ToLowerInvariant())
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

        private bool TryMapLanguagePairToTransliterator(DisplayLanguage sourceLanguage, DisplayLanguage targetLanguage, out TransliteratorMode mode)
        {
            mode = TransliteratorMode.EnglishToHebrew;

            if (sourceLanguage == targetLanguage)
            {
                return false;
            }

            if (sourceLanguage == DisplayLanguage.English && targetLanguage == DisplayLanguage.Hebrew)
            {
                mode = TransliteratorMode.EnglishToHebrew;
                return true;
            }

            if (sourceLanguage == DisplayLanguage.English && targetLanguage == DisplayLanguage.Russian)
            {
                mode = TransliteratorMode.EnglishToRussian;
                return true;
            }

            if (sourceLanguage == DisplayLanguage.Hebrew && targetLanguage == DisplayLanguage.English)
            {
                mode = TransliteratorMode.HebrewToEnglish;
                return true;
            }

            if (sourceLanguage == DisplayLanguage.Hebrew && targetLanguage == DisplayLanguage.Russian)
            {
                mode = TransliteratorMode.HebrewToRussian;
                return true;
            }

            if (sourceLanguage == DisplayLanguage.Russian && targetLanguage == DisplayLanguage.English)
            {
                mode = TransliteratorMode.RussianToEnglish;
                return true;
            }

            if (sourceLanguage == DisplayLanguage.Russian && targetLanguage == DisplayLanguage.Hebrew)
            {
                mode = TransliteratorMode.RussianToHebrew;
                return true;
            }

            return false;
        }
    }
}

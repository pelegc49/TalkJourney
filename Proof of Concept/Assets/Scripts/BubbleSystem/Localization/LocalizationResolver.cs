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

    [DisallowMultipleComponent]
    public class LocalizationResolver : MonoBehaviour, ILocalizationService
    {
        /// Event that fires whenever the display language changes.
        /// Subscribe to this to refresh UI text when the language changes mid-game.
        public static event System.Action OnLanguageChanged;

        [Header("Unity Localization")]
        [Tooltip("Default String Table Collection used when the key does not include an explicit table prefix.")]
        public string defaultStringTableCollection = "FirstScene";

        [Tooltip("When true, waits for LocalizationSettings initialization before first resolve.")]
        public bool waitForLocalizationInitialization = true;

        [Header("Fallback Behavior")]
        [Tooltip("If true, unresolved keys return the key itself. If false, unresolved keys return an empty string.")]
        public bool returnKeyWhenMissing = true;

        [Header("Display Language")]
        [Tooltip("Select the language to display. Changing this in the Inspector will immediately switch the language.")]
        public DisplayLanguage selectedLanguage = DisplayLanguage.English;

        private DisplayLanguage previousLanguage = DisplayLanguage.English;

        private void Awake()
        {
            EnsureLocalizationInitialized();
            ApplySelectedLanguage();
        }

        private void OnValidate()
        {
            // Apply language change immediately when edited in Inspector
            if (selectedLanguage != previousLanguage)
            {
                previousLanguage = selectedLanguage;
                #if UNITY_EDITOR
                if (!Application.isPlaying)
                {
                    // In editor, just update the locale without awaiting full initialization
                    string localeCode = EnumToLocaleCode(selectedLanguage);
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
                else if (Application.isPlaying)
                {
                    // During play mode, use the full method to trigger language change event
                    ApplySelectedLanguage();
                }
                #endif
            }
        }

        private void ApplySelectedLanguage()
        {
            string localeCode = EnumToLocaleCode(selectedLanguage);
            SetDisplayLanguage(localeCode);
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
                    Debug.Log($"Display language changed to: {localeCode} ({locale.name})");
                    
                    // Fire event to notify all listeners (bubbles) to refresh their text
                    OnLanguageChanged?.Invoke();
                    
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
    }
}

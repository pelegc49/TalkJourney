using System;
using UnityEngine;
using UnityEngine.Localization;
using UnityEngine.Localization.Settings;
using UnityEngine.Localization.Tables;

namespace TalkJourney.BubbleSystem.Localization
{
    [DisallowMultipleComponent]
    public class LocalizationResolver : MonoBehaviour, ILocalizationService
    {
        [Header("Unity Localization")]
        [Tooltip("Default String Table Collection used when the key does not include an explicit table prefix.")]
        public string defaultStringTableCollection = "FirstScene";

        [Tooltip("When true, waits for LocalizationSettings initialization before first resolve.")]
        public bool waitForLocalizationInitialization = true;

        [Header("Fallback Behavior")]
        [Tooltip("If true, unresolved keys return the key itself. If false, unresolved keys return an empty string.")]
        public bool returnKeyWhenMissing = true;

        private void Awake()
        {
            EnsureLocalizationInitialized();
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

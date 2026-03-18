using UnityEngine;
using UnityEngine.Localization.Settings;
using UnityEngine.UI;

namespace TalkJourney.BubbleSystem.Localization
{
    /// <summary>
    /// Provides a UI dropdown to switch languages. Automatically populates available locales.
    /// </summary>
    public class LanguageSwitcherUI : MonoBehaviour
    {
        [SerializeField]
        private Dropdown languageDropdown;

        [SerializeField]
        private LocalizationResolver localizationResolver;

        private void Awake()
        {
            if (localizationResolver == null)
            {
                localizationResolver = GetComponent<LocalizationResolver>();
            }

            if (localizationResolver == null)
            {
                localizationResolver = FindObjectOfType<LocalizationResolver>();
            }

            if (languageDropdown == null)
            {
                Debug.LogError("LanguageSwitcherUI: No Dropdown assigned. Please assign a Dropdown component.", this);
                return;
            }

            InitializeDropdown();
            languageDropdown.onValueChanged.AddListener(OnLanguageSelected);
        }

        private void InitializeDropdown()
        {
            languageDropdown.options.Clear();

            var availableLocales = LocalizationSettings.AvailableLocales;
            if (availableLocales?.Locales == null || availableLocales.Locales.Count == 0)
            {
                Debug.LogWarning("LanguageSwitcherUI: No locales are available.");
                return;
            }

            int selectedIndex = 0;
            var currentLocale = LocalizationSettings.SelectedLocale;

            for (int i = 0; i < availableLocales.Locales.Count; i++)
            {
                var locale = availableLocales.Locales[i];
                if (locale == null)
                    continue;

                // Display locale name (e.g., "English", "Hebrew", "Russian")
                // Falls back to locale code if name is not available
                string displayName = !string.IsNullOrEmpty(locale.name) ? locale.name : locale.Identifier.Code;

                languageDropdown.options.Add(new Dropdown.OptionData(displayName));

                // Mark current locale as selected
                if (currentLocale != null && locale.Identifier.Code == currentLocale.Identifier.Code)
                {
                    selectedIndex = i;
                }
            }

            languageDropdown.value = selectedIndex;
        }

        private void OnLanguageSelected(int index)
        {
            var availableLocales = LocalizationSettings.AvailableLocales;
            if (availableLocales?.Locales == null || index < 0 || index >= availableLocales.Locales.Count)
            {
                return;
            }

            var selectedLocale = availableLocales.Locales[index];
            if (selectedLocale == null)
            {
                return;
            }

            if (localizationResolver != null)
            {
                localizationResolver.SetDisplayLanguage(selectedLocale.Identifier.Code);
            }
            else
            {
                // Fallback: Set directly
                LocalizationSettings.SelectedLocale = selectedLocale;
                Debug.Log($"Language changed to: {selectedLocale.name}");
            }
        }

        private void OnDestroy()
        {
            if (languageDropdown != null)
            {
                languageDropdown.onValueChanged.RemoveListener(OnLanguageSelected);
            }
        }
    }
}

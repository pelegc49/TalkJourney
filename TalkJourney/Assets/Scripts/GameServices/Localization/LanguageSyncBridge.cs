using UnityEngine;

namespace TalkJourney.BubbleSystem.Localization
{
    /// <summary>
    /// Keeps LocalizationResolver display language and RealtimeWhisper language synchronized both ways.
    /// </summary>
    [DisallowMultipleComponent]
    public class LanguageSyncBridge : MonoBehaviour
    {
        [Tooltip("Localization resolver that controls display language.")]
        public LocalizationResolver localizationResolver;

        private RealtimeWhisper realtimeWhisper => RealtimeWhisper.Instance;

        private bool _isSyncing;

        private void Awake()
        {
            ResolveDependencies();
        }

        private void OnEnable()
        {
            ResolveDependencies();

            LocalizationResolver.OnDisplayLanguageChanged += OnLocalizationLanguageChanged;
            if (realtimeWhisper != null)
            {
                realtimeWhisper.LanguageChanged += OnWhisperLanguageChanged;
            }

            // Initial alignment when entering play mode: display language drives voice language.
            if (localizationResolver != null && realtimeWhisper != null)
            {
                SyncWhisperToLocalization(localizationResolver.selectedLanguage);
            }
        }

        private void OnDisable()
        {
            LocalizationResolver.OnDisplayLanguageChanged -= OnLocalizationLanguageChanged;
            if (realtimeWhisper != null)
            {
                realtimeWhisper.LanguageChanged -= OnWhisperLanguageChanged;
            }
        }

        private void ResolveDependencies()
        {
            if (localizationResolver == null)
            {
                localizationResolver = FindFirstObjectByType<LocalizationResolver>(FindObjectsInactive.Include);
            }
        }

        private void OnLocalizationLanguageChanged(DisplayLanguage language)
        {
            if (_isSyncing)
            {
                return;
            }

            SyncWhisperToLocalization(language);
        }

        private void OnWhisperLanguageChanged(RealtimeWhisper.Language language)
        {
            if (_isSyncing)
            {
                return;
            }

            if (localizationResolver == null)
            {
                return;
            }

            _isSyncing = true;
            localizationResolver.SetDisplayLanguage(MapWhisperToDisplay(language));
            _isSyncing = false;
        }

        private void SyncWhisperToLocalization(DisplayLanguage language)
        {
            if (realtimeWhisper == null)
            {
                return;
            }

            _isSyncing = true;
            realtimeWhisper.SetLanguage(MapDisplayToWhisper(language));
            _isSyncing = false;
        }

        private static RealtimeWhisper.Language MapDisplayToWhisper(DisplayLanguage language)
        {
            switch (language)
            {
                case DisplayLanguage.English:
                    return RealtimeWhisper.Language.English;
                case DisplayLanguage.Hebrew:
                    return RealtimeWhisper.Language.Hebrew;
                case DisplayLanguage.Russian:
                    return RealtimeWhisper.Language.Russian;
                default:
                    return RealtimeWhisper.Language.English;
            }
        }

        private static DisplayLanguage MapWhisperToDisplay(RealtimeWhisper.Language language)
        {
            switch (language)
            {
                case RealtimeWhisper.Language.English:
                    return DisplayLanguage.English;
                case RealtimeWhisper.Language.Hebrew:
                    return DisplayLanguage.Hebrew;
                case RealtimeWhisper.Language.Russian:
                    return DisplayLanguage.Russian;
                default:
                    return DisplayLanguage.English;
            }
        }
    }
}

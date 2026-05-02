using TMPro;
using UnityEngine;
using TalkJourney.BubbleSystem.Events;
using TalkJourney.BubbleSystem.Localization;

namespace TalkJourney.BubbleSystem.UI
{
    [DisallowMultipleComponent]
    public class SpeechResponseDisplay : MonoBehaviour
    {
        [Header("UI")]
        [Tooltip("Optional response root GameObject to show or hide. If empty, this GameObject is used.")]
        public GameObject responseRoot;

        [Tooltip("Text component used to display the recognized speech response.")]
        public TMP_Text responseText;

        [Tooltip("If enabled, hides the GameObject when the response is empty.")]
        public bool hideWhenEmpty = true;

        [Tooltip("If enabled, clears the response text when this component is disabled.")]
        public bool clearOnDisable = false;

        [Header("Localization")]
        [Tooltip("Optional LocalizationResolver. If empty, the script resolves one from the scene.")]
        public LocalizationResolver localizationResolver;

        private void Awake()
        {
            if (responseRoot == null)
            {
                responseRoot = gameObject;
            }

            if (responseText == null)
            {
                responseText = GetComponent<TMP_Text>();
            }

            if (responseText == null)
            {
                responseText = GetComponentInChildren<TMP_Text>(true);
            }
        }

        private void OnEnable()
        {
            BubbleEventBus.SpeechPhraseRecognized += OnSpeechPhraseRecognized;
            LocalizationResolver.OnLanguageChanged += OnLanguageChanged;
            ApplyTextDirection();
        }

        private void Start()
        {
            if (hideWhenEmpty && responseRoot != null && responseText != null && string.IsNullOrWhiteSpace(responseText.text))
            {
                responseText.text = "";
                responseRoot.SetActive(false);
            }
        }

        private void OnDisable()
        {
            BubbleEventBus.SpeechPhraseRecognized -= OnSpeechPhraseRecognized;
            LocalizationResolver.OnLanguageChanged -= OnLanguageChanged;

            if (clearOnDisable && responseText != null)
            {
                responseText.text = string.Empty;
            }
        }

        private void OnLanguageChanged()
        {
            ApplyTextDirection();
        }

        private void OnSpeechPhraseRecognized(string recognizedText)
        {
            if (responseText != null)
            {
                responseText.text = recognizedText ?? string.Empty;
                ApplyTextDirection();
            }

            if (hideWhenEmpty && responseRoot != null)
            {
                responseRoot.SetActive(!string.IsNullOrWhiteSpace(recognizedText));
            }
        }

        private void ApplyTextDirection()
        {
            if (responseText == null)
            {
                return;
            }

            var resolver = ResolveLocalizationResolver();
            if (resolver == null)
            {
                return;
            }

            responseText.isRightToLeftText = LocalizationResolver.IsRightToLeftLanguage(resolver.learningLanguage);
        }

        private LocalizationResolver ResolveLocalizationResolver()
        {
            if (localizationResolver != null)
            {
                return localizationResolver;
            }

            localizationResolver = FindFirstObjectByType<LocalizationResolver>(FindObjectsInactive.Include);
            return localizationResolver;
        }
    }
}
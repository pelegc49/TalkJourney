using UnityEngine;
using TalkJourney.BubbleSystem.Audio;
using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Events;
using TalkJourney.BubbleSystem.Interaction;
using TalkJourney.BubbleSystem.Localization;
using TMPro;

namespace TalkJourney.BubbleSystem.Bubbles
{
    [DisallowMultipleComponent]
    public class DisplayBubbleController : MonoBehaviour
    {
        [Header("Data")]
        public BubbleData bubbleData;

        [Header("Dependencies")]
        [Tooltip("Component implementing ILocalizationService.")]
        public MonoBehaviour localizationServiceBehaviour;

        [Tooltip("Component implementing IAudioPlaybackManager.")]
        public MonoBehaviour audioPlaybackManagerBehaviour;

        [Tooltip("Pointer interaction adapter for hover/click.")]
        public VrPointerInteractable interactable;

        [Header("Visuals")]
        [Tooltip("Primary localized word/phrase text.")]
        public TMP_Text primaryText;

        [Tooltip("Transliteration text shown only while hovering.")]
        public TMP_Text transliteratorText;

        [Header("Transliteration")]
        [Tooltip("Locale code used for transliteration values in Unity Localization, for example en-he or he-ru.")]
        public string transliteratorLocaleCode = "en-he";

        private ILocalizationService _localizationService;
        private IAudioPlaybackManager _audioPlaybackManager;

        private void Awake()
        {
            RefreshDependencies();
        }

        private void OnEnable()
        {
            if (interactable != null)
            {
                interactable.HoverEntered += OnHoverEntered;
                interactable.HoverExited += OnHoverExited;
                interactable.Clicked += OnClicked;
            }

            RefreshLocalizedTexts();
            SetTransliteratorVisible(false);
        }

        private void OnDisable()
        {
            if (interactable != null)
            {
                interactable.HoverEntered -= OnHoverEntered;
                interactable.HoverExited -= OnHoverExited;
                interactable.Clicked -= OnClicked;
            }
        }

        public void Initialize(BubbleData data)
        {
            bubbleData = data;
            RefreshLocalizedTexts();
            SetTransliteratorVisible(false);
        }

        public void RefreshDependencies()
        {
            _localizationService = localizationServiceBehaviour as ILocalizationService;
            _audioPlaybackManager = audioPlaybackManagerBehaviour as IAudioPlaybackManager;

            if (interactable == null)
            {
                interactable = GetComponent<VrPointerInteractable>();
            }

            if (_localizationService == null)
            {
                Debug.LogError("DisplayBubbleController requires localizationServiceBehaviour implementing ILocalizationService.", this);
            }

            if (_audioPlaybackManager == null)
            {
                Debug.LogError("DisplayBubbleController requires audioPlaybackManagerBehaviour implementing IAudioPlaybackManager.", this);
            }
        }

        public void RefreshLocalizedTexts()
        {
            if (bubbleData == null)
            {
                return;
            }

            if (primaryText != null)
            {
                primaryText.text = ResolveKey(bubbleData.primaryTextKey);
            }

            if (transliteratorText != null)
            {
                transliteratorText.text = ResolveTransliterator(bubbleData.primaryTextKey);
            }
        }

        private void OnHoverEntered()
        {
            BubbleEventBus.PublishBubbleHoverEntered(bubbleData);
            SetTransliteratorVisible(true);
        }

        private void OnHoverExited()
        {
            BubbleEventBus.PublishBubbleHoverExited(bubbleData);
            SetTransliteratorVisible(false);
        }

        private void OnClicked()
        {
            if (bubbleData == null || _audioPlaybackManager == null)
            {
                return;
            }

            BubbleEventBus.PublishBubbleClicked(bubbleData);
            _ = _audioPlaybackManager.PlayByIdentifierAsync(bubbleData.audioIdentifier);
        }

        private string ResolveKey(string key)
        {
            if (_localizationService == null || string.IsNullOrWhiteSpace(key))
            {
                return key ?? string.Empty;
            }

            return _localizationService.Resolve(key);
        }

        private string ResolveTransliterator(string key)
        {
            if (_localizationService == null || string.IsNullOrWhiteSpace(key))
            {
                return key ?? string.Empty;
            }

            if (!string.IsNullOrWhiteSpace(transliteratorLocaleCode)
                && _localizationService.TryResolveForLocaleCode(key, transliteratorLocaleCode, out var transliteratedValue))
            {
                return transliteratedValue;
            }

            return _localizationService.Resolve(key);
        }

        private void SetTransliteratorVisible(bool isVisible)
        {
            if (transliteratorText != null)
            {
                transliteratorText.enabled = isVisible;
            }
        }
    }
}

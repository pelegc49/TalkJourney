using UnityEngine;
using UnityEngine.UI;
using TalkJourney.BubbleSystem.Audio;
using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Events;
using TalkJourney.BubbleSystem.Interaction;
using TalkJourney.BubbleSystem.Localization;
using TalkJourney.GameServices;
using TMPro;
using System;

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
        [Obsolete("The transliterator locale code is now managed by LocalizationResolver. Use its selectedTransliterator dropdown instead.")]
        public string transliteratorLocaleCode = "en-he";

        [Header("Sizing")]
        [Tooltip("Minimum bubble size in pixels.")]
        public Vector2 minimumBubbleSize = new Vector2(50f, 50f);

        [Tooltip("Extra width/height padding added around the primary text in pixels.")]
        public Vector2 bubblePadding = new Vector2(32f, 16f);

        private ILocalizationService _localizationService;
        private IAudioPlaybackManager _audioPlaybackManager;
        private RectTransform _rectTransform;
        private LayoutElement _layoutElement;

        private void Awake()
        {
            _rectTransform = GetComponent<RectTransform>();
            _layoutElement = GetComponent<LayoutElement>();
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

            // Subscribe to language change events to refresh text when language switches
            LocalizationResolver.OnLanguageChanged += RefreshLocalizedTexts;

            // Subscribe to transliterator change events to refresh transliterator text
            LocalizationResolver.OnTransliteratorChanged += RefreshTransliteratorText;

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

            // Unsubscribe from language change events
            LocalizationResolver.OnLanguageChanged -= RefreshLocalizedTexts;

            // Unsubscribe from transliterator change events
            LocalizationResolver.OnTransliteratorChanged -= RefreshTransliteratorText;
        }

        public void Initialize(BubbleData data)
        {
            bubbleData = data;
            RefreshLocalizedTexts();
            SetTransliteratorVisible(false);
        }

        public void RefreshDependencies()
        {
            if (localizationServiceBehaviour == null)
            {
                var globalServices = GlobalGameServicesBootstrap.Instance;
                if (globalServices != null && globalServices.localizationResolver != null)
                {
                    localizationServiceBehaviour = globalServices.localizationResolver;
                }
                else
                {
                    localizationServiceBehaviour = GetComponentInParent<LocalizationResolver>(true);
                }
            }

            if (audioPlaybackManagerBehaviour == null)
            {
                audioPlaybackManagerBehaviour = GetComponentInParent<AudioPlaybackManager>(true);
            }

            _localizationService = localizationServiceBehaviour as ILocalizationService;
            _audioPlaybackManager = audioPlaybackManagerBehaviour as IAudioPlaybackManager;

            if (interactable == null)
            {
                interactable = GetComponent<VrPointerInteractable>();
            }

            if (_localizationService == null)
            {
                _localizationService = FindFirstObjectByType<LocalizationResolver>(FindObjectsInactive.Include);
            }

            if (_audioPlaybackManager == null)
            {
                _audioPlaybackManager = FindFirstObjectByType<AudioPlaybackManager>(FindObjectsInactive.Include);
            }

            if (_localizationService == null && bubbleData != null)
            {
                Debug.LogWarning("DisplayBubbleController could not resolve ILocalizationService. Text will fall back to keys.", this);
            }

            if (_audioPlaybackManager == null && bubbleData != null)
            {
                Debug.LogWarning("DisplayBubbleController could not resolve IAudioPlaybackManager. Bubble click audio will be disabled.", this);
            }
        }

        public void RefreshLocalizedTexts()
        {
            if (bubbleData == null)
            {
                return;
            }

            ApplyTextDirectionSettings();

            if (primaryText != null)
            {
                primaryText.text = ResolveKey(bubbleData.primaryTextKey);
                primaryText.ForceMeshUpdate();
            }

            if (transliteratorText != null)
            {
                transliteratorText.text = ResolveTransliterator(bubbleData.primaryTextKey);
                transliteratorText.ForceMeshUpdate();
            }

            ApplyPreferredBubbleSize();

            if (_rectTransform != null)
            {
                LayoutRebuilder.ForceRebuildLayoutImmediate(_rectTransform);
            }

            var parentRect = transform.parent as RectTransform;
            if (parentRect != null)
            {
                LayoutRebuilder.ForceRebuildLayoutImmediate(parentRect);
            }
        }

        private void ApplyPreferredBubbleSize()
        {
            if (_layoutElement == null)
            {
                _layoutElement = gameObject.AddComponent<LayoutElement>();
            }

            var preferredWidth = minimumBubbleSize.x;
            var preferredHeight = minimumBubbleSize.y;

            if (primaryText != null)
            {
                preferredWidth = Mathf.Max(preferredWidth, primaryText.preferredWidth + bubblePadding.x);
                preferredHeight = Mathf.Max(preferredHeight, primaryText.preferredHeight + bubblePadding.y);
            }

            _layoutElement.preferredWidth = preferredWidth;
            _layoutElement.preferredHeight = preferredHeight;
            _layoutElement.minWidth = minimumBubbleSize.x;
            _layoutElement.minHeight = minimumBubbleSize.y;
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

            var bubbleContent = ResolveKey(bubbleData.primaryTextKey);
            if (!string.IsNullOrWhiteSpace(bubbleContent))
            {
                _ = _audioPlaybackManager.PlayByTextAsync(bubbleContent);
            }
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

            string transliteratorCode = GetCurrentTransliteratorCode();
            if (!string.IsNullOrWhiteSpace(transliteratorCode)
                && _localizationService.TryResolveForLocaleCode(key, transliteratorCode, out var transliteratedValue))
            {
                return transliteratedValue;
            }

            return _localizationService.Resolve(key);
        }

        private void RefreshTransliteratorText()
        {
            if (bubbleData == null || transliteratorText == null)
            {
                return;
            }

            ApplyTextDirectionSettings();
            transliteratorText.text = ResolveTransliterator(bubbleData.primaryTextKey);
            transliteratorText.ForceMeshUpdate();
        }

        private void ApplyTextDirectionSettings()
        {
            var localizationResolver = ResolveLocalizationResolver();
            if (localizationResolver == null)
            {
                return;
            }

            if (primaryText != null)
            {
                primaryText.isRightToLeftText = LocalizationResolver.IsRightToLeftLanguage(localizationResolver.learningLanguage);
            }

            if (transliteratorText != null)
            {
                transliteratorText.isRightToLeftText = LocalizationResolver.IsRightToLeftLanguage(localizationResolver.nativeLanguage);
            }
        }

        private LocalizationResolver ResolveLocalizationResolver()
        {
            var localizationResolver = localizationServiceBehaviour as LocalizationResolver;
            if (localizationResolver != null)
            {
                return localizationResolver;
            }

            return FindFirstObjectByType<LocalizationResolver>(FindObjectsInactive.Include);
        }

        private string GetCurrentTransliteratorCode()
        {
            // Try to get the transliterator code from LocalizationResolver
            var localizationResolver = localizationServiceBehaviour as LocalizationResolver;
            if (localizationResolver != null)
            {
                return localizationResolver.GetCurrentTransliteratorCode();
            }

            // Fallback: try to find LocalizationResolver in scene
            localizationResolver = FindFirstObjectByType<LocalizationResolver>();
            if (localizationResolver != null)
            {
                return localizationResolver.GetCurrentTransliteratorCode();
            }

            // Last resort: use the deprecated field
            #pragma warning disable CS0618
            return transliteratorLocaleCode;
            #pragma warning restore CS0618
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

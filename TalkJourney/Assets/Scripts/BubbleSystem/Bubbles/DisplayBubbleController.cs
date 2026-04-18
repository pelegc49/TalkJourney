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
using UnityEngine.Serialization;

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

        [Tooltip("Root object of TransliteratorText_UI shown only while hovering.")]
        public GameObject transliteratorTextObject;

        [Header("Transliteration")]
        [Tooltip("Locale code used for transliteration values in Unity Localization, for example en-he or he-ru.")]
        [Obsolete("The transliterator locale code is now managed by LocalizationResolver. Use its selectedTransliterator dropdown instead.")]
        public string transliteratorLocaleCode = "en-he";

        [Header("Sizing")]
        [Tooltip("Minimum bubble size in pixels.")]
        public Vector2 minimumBubbleSize = new Vector2(50f, 50f);

        [Tooltip("Extra width/height padding added around the primary text in pixels.")]
        public Vector2 bubblePadding = new Vector2(32f, 16f);

        [Tooltip("When enabled, PrimaryText_UI RectTransform is resized to the exact preferred text size and DisplayBubble is matched 1:1 to it.")]
        public bool matchPrimaryTextAndDisplayBubbleToContent = true;

        [Tooltip("Optional explicit RectTransform for PrimaryText_UI. If empty, primaryText.rectTransform is used.")]
        public RectTransform primaryTextRectTransform;

        [FormerlySerializedAs("primaryTextMargin")]
        [Tooltip("Extra width/height padding added to PrimaryText_UI around its content.")]
        public Vector2 primaryTextPadding = Vector2.zero;

        [Tooltip("Optional explicit RectTransform for DisplayBubble image. If empty, it is auto-resolved under PrimaryText_UI/DisplayBubble.")]
        public RectTransform primaryDisplayBubbleRectTransform;

        [FormerlySerializedAs("primaryDisplayBubbleMargin")]
        [Tooltip("Extra width/height padding added to DisplayBubble around PrimaryText_UI.")]
        public Vector2 primaryDisplayBubblePadding = Vector2.zero;

        [Tooltip("Optional explicit RectTransform for TransliteratorText_UI. If empty, TMP_Text on transliteratorTextObject is used.")]
        public RectTransform transliteratorTextRectTransform;

        [FormerlySerializedAs("transliteratorTextMargin")]
        [Tooltip("Extra width/height padding added to TransliteratorText_UI around its content.")]
        public Vector2 transliteratorTextPadding = Vector2.zero;

        [Tooltip("Optional explicit RectTransform for TransliteratorBubble image. If empty, it is auto-resolved under TransliteratorText_UI/TransliteratorBubble.")]
        public RectTransform transliteratorBubbleRectTransform;

        [FormerlySerializedAs("transliteratorBubbleMargin")]
        [Tooltip("Extra width/height padding added to TransliteratorBubble around TransliteratorText_UI.")]
        public Vector2 transliteratorBubblePadding = Vector2.zero;

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

            var transliteratorText = ResolveTransliteratorTextComponent();
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

            if (matchPrimaryTextAndDisplayBubbleToContent)
            {
                var primarySize = ResizePrimaryTextAndDisplayBubbleToContent();
                var transliteratorSize = ResizeTransliteratorTextAndBubbleToContent();
                var finalWidth = primarySize.x; //Mathf.Max(primarySize.x, transliteratorSize.x);
                var finalHeight = primarySize.y; //Mathf.Max(primarySize.y, transliteratorSize.y);
                _layoutElement.preferredWidth = finalWidth;
                _layoutElement.preferredHeight = finalHeight;
                _layoutElement.minWidth = finalWidth;
                _layoutElement.minHeight = finalHeight;
                return;
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

        private Vector2 ResizePrimaryTextAndDisplayBubbleToContent()
        {
            var primaryRect = ResolvePrimaryTextRectTransform();
            if (primaryText == null || primaryRect == null)
            {
                return minimumBubbleSize;
            }

            var preferred = primaryText.GetPreferredValues(primaryText.text, Mathf.Infinity, Mathf.Infinity);
            var textWidth = Mathf.Max(0f, preferred.x + primaryTextPadding.x);
            var textHeight = Mathf.Max(0f, preferred.y + primaryTextPadding.y);

            primaryRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, textWidth);
            primaryRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, textHeight);

            var bubbleWidth = Mathf.Max(0f, textWidth + primaryDisplayBubblePadding.x);
            var bubbleHeight = Mathf.Max(0f, textHeight + primaryDisplayBubblePadding.y);

            var displayBubbleRect = ResolvePrimaryDisplayBubbleRectTransform(primaryRect);
            if (displayBubbleRect != null)
            {
                displayBubbleRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, bubbleWidth);
                displayBubbleRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, bubbleHeight);
            }

            return new Vector2(Mathf.Max(textWidth, bubbleWidth), Mathf.Max(textHeight, bubbleHeight));
        }

        private RectTransform ResolvePrimaryTextRectTransform()
        {
            if (primaryTextRectTransform != null)
            {
                return primaryTextRectTransform;
            }

            if (primaryText != null)
            {
                return primaryText.rectTransform;
            }

            return null;
        }

        private RectTransform ResolvePrimaryDisplayBubbleRectTransform(RectTransform primaryRect)
        {
            if (primaryDisplayBubbleRectTransform != null)
            {
                return primaryDisplayBubbleRectTransform;
            }

            if (primaryRect == null)
            {
                return null;
            }

            var displayBubble = primaryRect.Find("DisplayBubble") as RectTransform;
            if (displayBubble != null)
            {
                return displayBubble;
            }

            var image = primaryRect.GetComponentInChildren<Image>(true);
            return image != null ? image.rectTransform : null;
        }

        private Vector2 ResizeTransliteratorTextAndBubbleToContent()
        {
            var transliteratorText = ResolveTransliteratorTextComponent();
            var transliteratorRect = ResolveTransliteratorTextRectTransform();
            if (transliteratorText == null || transliteratorRect == null)
            {
                return Vector2.zero;
            }

            var preferred = transliteratorText.GetPreferredValues(transliteratorText.text, Mathf.Infinity, Mathf.Infinity);
            var textWidth = Mathf.Max(0f, preferred.x + transliteratorTextPadding.x);
            var textHeight = Mathf.Max(0f, preferred.y + transliteratorTextPadding.y);

            transliteratorRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, textWidth);
            transliteratorRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, textHeight);

            var bubbleWidth = Mathf.Max(0f, textWidth + transliteratorBubblePadding.x);
            var bubbleHeight = Mathf.Max(0f, textHeight + transliteratorBubblePadding.y);

            var transliteratorBubbleRect = ResolveTransliteratorBubbleRectTransform(transliteratorRect);
            if (transliteratorBubbleRect != null)
            {
                transliteratorBubbleRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, bubbleWidth);
                transliteratorBubbleRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, bubbleHeight);
            }

            return new Vector2(Mathf.Max(textWidth, bubbleWidth), Mathf.Max(textHeight, bubbleHeight));
        }

        private RectTransform ResolveTransliteratorTextRectTransform()
        {
            if (transliteratorTextRectTransform != null)
            {
                return transliteratorTextRectTransform;
            }

            var transliteratorText = ResolveTransliteratorTextComponent();
            if (transliteratorText != null)
            {
                return transliteratorText.rectTransform;
            }

            return null;
        }

        private TMP_Text ResolveTransliteratorTextComponent()
        {
            if (transliteratorTextObject == null)
            {
                return null;
            }

            return transliteratorTextObject.GetComponent<TMP_Text>();
        }

        private RectTransform ResolveTransliteratorBubbleRectTransform(RectTransform transliteratorRect)
        {
            if (transliteratorBubbleRectTransform != null)
            {
                return transliteratorBubbleRectTransform;
            }

            if (transliteratorRect == null)
            {
                return null;
            }

            var transliteratorBubble = transliteratorRect.Find("TransliteratorBubble") as RectTransform;
            if (transliteratorBubble != null)
            {
                return transliteratorBubble;
            }

            var image = transliteratorRect.GetComponentInChildren<Image>(true);
            return image != null ? image.rectTransform : null;
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
            var transliteratorText = ResolveTransliteratorTextComponent();
            if (bubbleData == null || transliteratorText == null)
            {
                return;
            }

            ApplyTextDirectionSettings();
            transliteratorText.text = ResolveTransliterator(bubbleData.primaryTextKey);
            transliteratorText.ForceMeshUpdate();

            if (matchPrimaryTextAndDisplayBubbleToContent)
            {
                ApplyPreferredBubbleSize();
            }
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

            var transliteratorText = ResolveTransliteratorTextComponent();
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
            if (transliteratorTextObject != null)
            {
                transliteratorTextObject.SetActive(isVisible);
            }

            var transliteratorRect = ResolveTransliteratorTextRectTransform();
            var transliteratorBubbleRect = ResolveTransliteratorBubbleRectTransform(transliteratorRect);
            if (transliteratorBubbleRect != null && transliteratorBubbleRect.gameObject != transliteratorTextObject)
            {
                transliteratorBubbleRect.gameObject.SetActive(isVisible);
            }
        }

    }
}

using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using TalkJourney.BubbleSystem.Audio;
using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Interaction;
using TalkJourney.BubbleSystem.Localization;

namespace TalkJourney.BubbleSystem.Bubbles
{
    public enum SentencePlaybackMode
    {
        SequentialBubbles,
        PreferFullSentenceClip
    }

    [DisallowMultipleComponent]
    public class SentenceBubbleController : MonoBehaviour
    {
        [Header("Dependencies")]
        [Tooltip("Component implementing IAudioPlaybackManager.")]
        public MonoBehaviour audioPlaybackManagerBehaviour;

        [Tooltip("Component implementing ILocalizationService; injected into spawned display bubbles.")]
        public MonoBehaviour localizationServiceBehaviour;

        [Header("Composition")]
        [Tooltip("Parent transform where display bubbles are instantiated.")]
        public Transform sentenceBubbleParent;

        [Tooltip("If enabled, sentenceBubbleParent will be configured to use FlowLayoutGroup at runtime.")]
        public bool autoConfigureSentenceAreaLayout = true;

        [Tooltip("Fallback prefab if BubbleData.visualElementPrefab is not set.")]
        public DisplayBubbleController fallbackDisplayBubblePrefab;

        [Header("Speaker Button")]
        [Tooltip("Interactable speaker button shown near the sentence bubble.")]
        public VrPointerInteractable speakerButtonInteractable;

        [Tooltip("Optional RectTransform used as sentence area bounds. Defaults to sentenceBubbleParent as RectTransform.")]
        public RectTransform sentenceAreaRect;

        [Tooltip("Fixed world-space offset from the right edge of SentenceArea to place the speaker button.")]
        public Vector3 speakerButtonOffsetFromSentenceArea = new Vector3(0.25f, 0f, 0f);

        public SentencePlaybackMode playbackMode = SentencePlaybackMode.PreferFullSentenceClip;

        private readonly List<DisplayBubbleController> _spawnedBubbles = new List<DisplayBubbleController>();
        private IAudioPlaybackManager _audioPlaybackManager;
        private StageData _activeStage;

        private void Awake()
        {
            RefreshDependencies();
            // EnsureSentenceAreaLayout();
            SetSpeakerButtonVisible(false);
        }

        private void OnEnable()
        {
            if (speakerButtonInteractable != null)
            {
                speakerButtonInteractable.Clicked += OnSpeakerClicked;
            }

            // Subscribe to language change events to refresh bubbles when language switches
            LocalizationResolver.OnLanguageChanged += RefreshBubblesForLanguageChange;
        }

        private void OnDisable()
        {
            if (speakerButtonInteractable != null)
            {
                speakerButtonInteractable.Clicked -= OnSpeakerClicked;
            }

            // Unsubscribe from language change events
            LocalizationResolver.OnLanguageChanged -= RefreshBubblesForLanguageChange;
        }

        private void RefreshBubblesForLanguageChange()
        {
            // Refresh all spawned display bubbles when language changes
            foreach (var bubble in _spawnedBubbles)
            {
                if (bubble != null)
                {
                    bubble.RefreshLocalizedTexts();
                }
            }
        }

        public void LoadStage(StageData stageData)
        {
            _activeStage = stageData;
            RebuildSentence(stageData);

            var shouldShowSpeaker = stageData != null;
            SetSpeakerButtonVisible(shouldShowSpeaker);
            if (shouldShowSpeaker)
            {
                RepositionSpeakerButton();
            }
        }

        public void RebuildSentence(StageData stageData)
        {
            ClearSpawnedBubbles();

            // EnsureSentenceAreaLayout();

            if (stageData == null || sentenceBubbleParent == null)
            {
                return;
            }

            for (int i = 0; i < stageData.sentenceBubbles.Count; i++)
            {
                var bubbleData = stageData.sentenceBubbles[i];
                if (bubbleData == null)
                {
                    continue;
                }

                var controller = SpawnBubbleController(bubbleData);
                if (controller == null)
                {
                    continue;
                }

                controller.Initialize(bubbleData);
                _spawnedBubbles.Add(controller);
            }

            var sentenceRect = sentenceBubbleParent as RectTransform;
            if (sentenceRect != null)
            {
                LayoutRebuilder.ForceRebuildLayoutImmediate(sentenceRect);
            }

            RepositionSpeakerButton();
        }

        public void RefreshDependencies()
        {
            _audioPlaybackManager = audioPlaybackManagerBehaviour as IAudioPlaybackManager;
            if (_audioPlaybackManager == null)
            {
                Debug.LogError("SentenceBubbleController requires audioPlaybackManagerBehaviour implementing IAudioPlaybackManager.", this);
            }
        }

        public async Task PlaySentenceAsync(CancellationToken cancellationToken = default)
        {
            if (_audioPlaybackManager == null || _activeStage == null)
            {
                return;
            }

            if (playbackMode == SentencePlaybackMode.PreferFullSentenceClip
                && !string.IsNullOrWhiteSpace(_activeStage.fullSentenceAudioIdentifier))
            {
                var played = await _audioPlaybackManager.PlayByIdentifierAsync(_activeStage.fullSentenceAudioIdentifier, cancellationToken);
                if (played)
                {
                    return;
                }
            }

            var identifiers = new List<string>(_activeStage.sentenceBubbles.Count);
            for (int i = 0; i < _activeStage.sentenceBubbles.Count; i++)
            {
                var bubble = _activeStage.sentenceBubbles[i];
                if (bubble != null && !string.IsNullOrWhiteSpace(bubble.audioIdentifier))
                {
                    identifiers.Add(bubble.audioIdentifier);
                }
            }

            await _audioPlaybackManager.PlaySequenceAsync(identifiers, cancellationToken);
        }

        private void OnSpeakerClicked()
        {
            _ = PlaySentenceAsync();
        }

        private DisplayBubbleController SpawnBubbleController(BubbleData bubbleData)
        {
            var prefab = ResolvePrefab(bubbleData);
            if (prefab == null)
            {
                return null;
            }

            var instance = Instantiate(prefab, sentenceBubbleParent);
            instance.localizationServiceBehaviour = localizationServiceBehaviour;
            instance.audioPlaybackManagerBehaviour = audioPlaybackManagerBehaviour;
            instance.RefreshDependencies();
            return instance;
        }

        private DisplayBubbleController ResolvePrefab(BubbleData bubbleData)
        {
            if (bubbleData != null && bubbleData.visualElementPrefab != null)
            {
                return bubbleData.visualElementPrefab.GetComponent<DisplayBubbleController>();
            }

            return fallbackDisplayBubblePrefab;
        }

        private void ClearSpawnedBubbles()
        {
            for (int i = 0; i < _spawnedBubbles.Count; i++)
            {
                var spawned = _spawnedBubbles[i];
                if (spawned != null)
                {
                    Destroy(spawned.gameObject);
                }
            }

            _spawnedBubbles.Clear();
        }

        private void SetSpeakerButtonVisible(bool isVisible)
        {
            var speakerTransform = ResolveSpeakerButtonTransform();
            if (speakerTransform != null)
            {
                speakerTransform.gameObject.SetActive(isVisible);
            }
        }

        private Transform ResolveSpeakerButtonTransform()
        {
            if (speakerButtonInteractable != null)
            {
                return speakerButtonInteractable.transform;
            }

            return null;
        }

        private RectTransform ResolveSentenceAreaRect()
        {
            if (sentenceAreaRect != null)
            {
                return sentenceAreaRect;
            }

            return sentenceBubbleParent as RectTransform;
        }

        private void RepositionSpeakerButton()
        {
            var speakerTransform = ResolveSpeakerButtonTransform();
            var areaRect = ResolveSentenceAreaRect();
            if (speakerTransform == null || areaRect == null)
            {
                return;
            }

            var corners = new Vector3[4];
            areaRect.GetWorldCorners(corners);

            // 2 = top-right, 3 = bottom-right. Midpoint gives the vertical center of right edge.
            var rightEdgeCenter = (corners[2] + corners[3]) * 0.5f;

            // Keep local orientation independent from sentence text length by pinning to sentence area bounds.
            speakerTransform.position = rightEdgeCenter + speakerButtonOffsetFromSentenceArea;
        }
    }
}

using System.Collections;
using UnityEngine;
using TalkJourney.BubbleSystem.Audio;
using TalkJourney.BubbleSystem.Bubbles;
using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Events;
using TalkJourney.BubbleSystem.Interaction;
using TalkJourney.BubbleSystem.Localization;
using TalkJourney.BubbleSystem.Speech;
using TalkJourney.GameServices;
using UnityEngine.UI;

namespace TalkJourney.BubbleSystem.Flow
{
    [DefaultExecutionOrder(-1000)]
    [DisallowMultipleComponent]
    public class BubbleSystemBootstrap : MonoBehaviour
    {
        [Header("Scene Data")]
        public StageData initialStage;
        public Transform sentenceBubbleParent;
        public Transform selectionBubbleParent;
        public DisplayBubbleController fallbackDisplayBubblePrefab;
        public SelectionBubbleController selectionBubblePrefab;
        public VrPointerInteractable speakerButtonInteractable;
        public VrPointerInteractable bypassButtonInteractable;

        [Header("Layout")]
        [Min(0f)]
        [Tooltip("Vertical spacing in pixels between SentenceArea and SelectionArea.")]
        public float sentenceSelectionSpacing = 100f;

        [Header("External Services")]
        [Tooltip("Optional speech recognition component implementing ISpeechRecognitionService. If empty, GlobalGameServicesBootstrap.realtimeWhisper is used first.")]
        public MonoBehaviour speechRecognitionBehaviour;

        [Tooltip("Optional localization component implementing ILocalizationService. If empty, scene-wide service resolution is used.")]
        public MonoBehaviour localizationServiceBehaviour;

        [Tooltip("Optional audio backend component implementing IAudioBackendClient. If empty, scene-wide service resolution is used.")]
        public MonoBehaviour audioBackendClientBehaviour;

        [Tooltip("Optional playback source. If empty, an AudioSource is added here.")]
        public AudioSource playbackSource;

        [Header("Service Resolution")]
        [Tooltip("Services are resolved from GlobalGameServicesBootstrap. If not found there, scene-wide fallback is used. Local service creation is disabled by design.")]
        public bool strictGlobalServiceMode = true;

        [Header("Auto Setup")]
        [Tooltip("When enabled, setup runs in Awake. Disable if this bootstrap is instantiated/started by a launcher on demand.")]
        public bool initializeOnAwake = true;

        public bool autoFindSpeechRecognitionBehaviour = true;
        public bool autoCreateAudioSource = true;

        [Tooltip("When enabled, SentenceArea and SelectionArea heights are recalculated from wrapped child rows every frame during play mode.")]
        public bool autoResizeAreasByRows = true;

        [Header("Animation")]
        [Tooltip("Optional animator used to play the show/hide animations on the BubbleSystemBootstrap.")]
        public Animator bootstrapAnimator;

        [Tooltip("Animator bool parameter that triggers the show animation when this object becomes visible.")]
        public string showAnimationBool = "IsVisible";

        [Tooltip("Animator trigger parameter that triggers the show animation when this object becomes visible.")]
        public string showAnimationTrigger = "FadeIn";

        [Tooltip("Animator bool parameter that triggers the hide animation on journey completion.")]
        public string hideAnimationBool = "IsVisible";

        [Tooltip("Animator trigger parameter that triggers the hide animation on journey completion.")]
        public string hideAnimationTrigger = "FadeOut";

        [Tooltip("Duration of the hide animation after journey completion.")]
        public float hideAnimationDuration = 0.5f;

        private bool _isSetupComplete;
        private Coroutine _hideCompletionCoroutine;
        private int _lastSentenceChildCount = -1;
        private int _lastSelectionChildCount = -1;
        private float _lastSentenceWidth = -1f;
        private float _lastSelectionWidth = -1f;

        private void Reset()
        {
            EnsureSetup();
        }

        private void Awake()
        {
            if (initializeOnAwake)
            {
                EnsureSetup();
            }
        }

        private void OnEnable()
        {
            PlayShowAnimation();
            BubbleEventBus.JourneyCompleted += OnJourneyCompleted;
        }

        private void OnDisable()
        {
            ResetShowAnimationState();
            BubbleEventBus.JourneyCompleted -= OnJourneyCompleted;
        }

        private void OnValidate()
        {
            if (!Application.isPlaying)
            {
                CenterSentenceAndSelectionAreas();
            }
        }

        private void PlayShowAnimation()
        {
            var animator = bootstrapAnimator != null ? bootstrapAnimator : GetComponent<Animator>();
            if (animator == null)
            {
                return;
            }

            if (!string.IsNullOrEmpty(showAnimationTrigger))
            {
                animator.SetTrigger(showAnimationTrigger);
                return;
            }

            if (!string.IsNullOrEmpty(showAnimationBool))
            {
                animator.SetBool(showAnimationBool, true);
            }
        }

        private void OnJourneyCompleted(StageData _, SelectionBubbleData __)
        {
            if (_hideCompletionCoroutine != null)
            {
                return;
            }

            _hideCompletionCoroutine = StartCoroutine(HideOnJourneyCompletedRoutine());
        }

        private System.Collections.IEnumerator HideOnJourneyCompletedRoutine()
        {
            var animator = bootstrapAnimator != null ? bootstrapAnimator : GetComponent<Animator>();
            if (animator != null)
            {
                if (!string.IsNullOrEmpty(hideAnimationTrigger))
                {
                    animator.SetTrigger(hideAnimationTrigger);
                }
                else if (!string.IsNullOrEmpty(hideAnimationBool))
                {
                    animator.SetBool(hideAnimationBool, false);
                }
            }

            yield return new WaitForSeconds(Mathf.Max(0f, hideAnimationDuration));

            gameObject.SetActive(false);
            BubbleEventBus.PublishBubbleSystemHidden();
            _hideCompletionCoroutine = null;
        }

        private void ResetShowAnimationState()
        {
            var animator = bootstrapAnimator != null ? bootstrapAnimator : GetComponent<Animator>();
            if (animator == null || string.IsNullOrEmpty(showAnimationBool))
            {
                return;
            }

            animator.SetBool(showAnimationBool, false);
        }

        private void LateUpdate()
        {
            if (!Application.isPlaying || !autoResizeAreasByRows)
            {
                return;
            }

            var sentenceRect = sentenceBubbleParent as RectTransform;
            var selectionRect = selectionBubbleParent as RectTransform;
            if (sentenceRect == null || selectionRect == null)
            {
                return;
            }

            var sentenceChildCount = CountActiveChildren(sentenceRect);
            var selectionChildCount = CountActiveChildren(selectionRect);
            var sentenceWidth = sentenceRect.rect.width;
            var selectionWidth = selectionRect.rect.width;

            var hasChanged = sentenceChildCount != _lastSentenceChildCount
                || selectionChildCount != _lastSelectionChildCount
                || !Mathf.Approximately(sentenceWidth, _lastSentenceWidth)
                || !Mathf.Approximately(selectionWidth, _lastSelectionWidth);

            if (!hasChanged)
            {
                return;
            }

            _lastSentenceChildCount = sentenceChildCount;
            _lastSelectionChildCount = selectionChildCount;
            _lastSentenceWidth = sentenceWidth;
            _lastSelectionWidth = selectionWidth;

            CenterSentenceAndSelectionAreas();
        }

        public void EnsureSetup()
        {
            if (_isSetupComplete)
            {
                return;
            }

            var stageController = GetOrAddComponent<StageController>();
            var sentenceBubbleController = GetOrAddComponent<SentenceBubbleController>();
            var selectionSpeechMatcher = GetOrAddComponent<SelectionSpeechMatcher>();
            var audioPlaybackManager = GetOrAddComponent<AudioPlaybackManager>();

            localizationServiceBehaviour = EnsureLocalizationService();
            audioBackendClientBehaviour = EnsureAudioBackendClient();

            if (autoCreateAudioSource && playbackSource == null)
            {
                playbackSource = GetComponent<AudioSource>();
                if (playbackSource == null)
                {
                    playbackSource = gameObject.AddComponent<AudioSource>();
                }
            }

            speechRecognitionBehaviour = EnsureSpeechRecognitionService(selectionSpeechMatcher);

            stageController.initialStage = initialStage;
            stageController.sentenceBubbleController = sentenceBubbleController;
            stageController.selectionBubbleParent = selectionBubbleParent;
            stageController.selectionBubblePrefab = selectionBubblePrefab;
            stageController.selectionSpeechMatcher = selectionSpeechMatcher;
            stageController.localizationServiceBehaviour = localizationServiceBehaviour;
            stageController.audioPlaybackManagerBehaviour = audioPlaybackManager;

            sentenceBubbleController.audioPlaybackManagerBehaviour = audioPlaybackManager;
            sentenceBubbleController.localizationServiceBehaviour = localizationServiceBehaviour;
            sentenceBubbleController.sentenceBubbleParent = sentenceBubbleParent;
            sentenceBubbleController.fallbackDisplayBubblePrefab = fallbackDisplayBubblePrefab;
            sentenceBubbleController.speakerButtonInteractable = speakerButtonInteractable;
            sentenceBubbleController.RefreshDependencies();

            selectionSpeechMatcher.speechRecognitionBehaviour = speechRecognitionBehaviour;
            selectionSpeechMatcher.bypassButtonInteractable = bypassButtonInteractable;
            selectionSpeechMatcher.RefreshDependencies();

            audioPlaybackManager.playbackSource = playbackSource;
            audioPlaybackManager.backendClientBehaviour = audioBackendClientBehaviour;
            audioPlaybackManager.RefreshDependencies();

            CenterSentenceAndSelectionAreas();

            _isSetupComplete = true;
        }

        /// <summary>
        /// Starts BubbleSystem for a specific stage. Useful for menu buttons (Level 1, Level 2, ...).
        /// </summary>
        public void StartStage(StageData stageData)
        {
            if (stageData == null)
            {
                Debug.LogWarning("BubbleSystemBootstrap.StartStage was called with null stageData.", this);
                return;
            }

            EnsureSetup();

            var stageController = GetComponent<StageController>();
            if (stageController == null)
            {
                Debug.LogError("BubbleSystemBootstrap could not find StageController after setup.", this);
                return;
            }

            stageController.autoStartInitialStage = false;
            stageController.LoadStage(stageData);
        }

        /// <summary>
        /// Starts BubbleSystem using the configured initialStage field.
        /// </summary>
        public void StartInitialStage()
        {
            if (initialStage == null)
            {
                Debug.LogWarning("BubbleSystemBootstrap.StartInitialStage was called but initialStage is not assigned.", this);
                return;
            }

            StartStage(initialStage);
        }

        private MonoBehaviour EnsureLocalizationService()
        {
            if (localizationServiceBehaviour != null)
            {
                return localizationServiceBehaviour;
            }

            // Priority 1: Try to get from GlobalGameServicesBootstrap (persistent global services)
            var globalServices = GlobalGameServicesBootstrap.Instance;
            if (globalServices != null && globalServices.localizationResolver != null)
            {
                localizationServiceBehaviour = globalServices.localizationResolver;
                return localizationServiceBehaviour;
            }

            // Priority 2: Try to find in scene if globals not available
            localizationServiceBehaviour = FindFirstSceneBehaviourImplementing<ILocalizationService>(this);
            if (localizationServiceBehaviour != null)
            {
                return localizationServiceBehaviour;
            }

            // Priority 3: Try to get from this GameObject if already added
            localizationServiceBehaviour = GetComponent<LocalizationResolver>();
            if (localizationServiceBehaviour != null)
            {
                return localizationServiceBehaviour;
            }

            // Strict mode: Fail if no service found. Local creation is disabled by design.
            Debug.LogError(
                "BubbleSystemBootstrap could not resolve ILocalizationService. " +
                "Ensure GlobalGameServicesBootstrap is set up in a persistent scene with localizationResolver assigned, or " +
                "add a LocalizationResolver to this scene before BubbleSystemBootstrap initializes.",
                this);

            return null;
        }

        private MonoBehaviour EnsureAudioBackendClient()
        {
            if (audioBackendClientBehaviour != null)
            {
                return audioBackendClientBehaviour;
            }

            // Priority 1: Try to get from GlobalGameServicesBootstrap (persistent global services)
            var globalServices = GlobalGameServicesBootstrap.Instance;
            if (globalServices != null && globalServices.audioBackendClient != null)
            {
                audioBackendClientBehaviour = globalServices.audioBackendClient;
                return audioBackendClientBehaviour;
            }

            // Priority 2: Try to find in scene if globals not available
            audioBackendClientBehaviour = FindFirstSceneBehaviourImplementing<IAudioBackendClient>(this);
            if (audioBackendClientBehaviour != null)
            {
                return audioBackendClientBehaviour;
            }

            // Priority 3: Try to get from this GameObject if already added
            audioBackendClientBehaviour = GetComponent<AudioBackendClient>();
            if (audioBackendClientBehaviour != null)
            {
                return audioBackendClientBehaviour;
            }

            // Strict mode: Fail if no service found. Local creation is disabled by design.
            Debug.LogError(
                "BubbleSystemBootstrap could not resolve IAudioBackendClient. " +
                "Ensure GlobalGameServicesBootstrap is set up in a persistent scene with audioBackendClient assigned, or " +
                "add an AudioBackendClient to this scene before BubbleSystemBootstrap initializes.",
                this);

            return null;
        }

        private MonoBehaviour FindFirstSceneBehaviourImplementing<T>(Component excludedComponent) where T : class
        {
            var sceneBehaviours = FindObjectsOfType<MonoBehaviour>(true);
            for (int i = 0; i < sceneBehaviours.Length; i++)
            {
                var behaviour = sceneBehaviours[i];
                if (behaviour == null || behaviour == excludedComponent)
                {
                    continue;
                }

                if (behaviour is T)
                {
                    return behaviour;
                }
            }

            return null;
        }

        private T GetOrAddComponent<T>() where T : Component
        {
            var component = GetComponent<T>();
            if (component == null)
            {
                component = gameObject.AddComponent<T>();
            }

            return component;
        }

        private MonoBehaviour EnsureSpeechRecognitionService(Component excludedComponent)
        {
            if (speechRecognitionBehaviour != null)
            {
                return speechRecognitionBehaviour;
            }

            // Priority 1: Use RealtimeWhisper singleton
            var realtimeWhisper = RealtimeWhisper.Instance;
            if (realtimeWhisper != null && realtimeWhisper is ISpeechRecognitionService speechService)
            {
                speechRecognitionBehaviour = (MonoBehaviour)speechService;
                return speechRecognitionBehaviour;
            }

            if (!autoFindSpeechRecognitionBehaviour)
            {
                return null;
            }

            // Priority 2: Try local/scene-wide lookup for backward compatibility
            speechRecognitionBehaviour = FindFirstBehaviourImplementing<ISpeechRecognitionService>(excludedComponent);
            if (speechRecognitionBehaviour != null)
            {
                return speechRecognitionBehaviour;
            }

            Debug.LogError(
                "BubbleSystemBootstrap could not resolve ISpeechRecognitionService. " +
                "Ensure RealtimeWhisper component is present in the scene.",
                this);

            return null;
        }

        private MonoBehaviour FindFirstBehaviourImplementing<T>(Component excludedComponent) where T : class
        {
            var localBehaviours = GetComponents<MonoBehaviour>();
            for (int i = 0; i < localBehaviours.Length; i++)
            {
                var behaviour = localBehaviours[i];
                if (behaviour == null || behaviour == excludedComponent)
                {
                    continue;
                }

                if (behaviour is T)
                {
                    return behaviour;
                }
            }

            var sceneBehaviours = FindObjectsOfType<MonoBehaviour>(true);
            for (int i = 0; i < sceneBehaviours.Length; i++)
            {
                var behaviour = sceneBehaviours[i];
                if (behaviour == null || behaviour == excludedComponent)
                {
                    continue;
                }

                if (behaviour is T)
                {
                    return behaviour;
                }
            }

            return null;
        }

        private void CenterSentenceAndSelectionAreas()
        {
            var sentenceRect = sentenceBubbleParent as RectTransform;
            var selectionRect = selectionBubbleParent as RectTransform;
            if (sentenceRect == null || selectionRect == null)
            {
                return;
            }

            ApplyHeightFromRows(sentenceRect);
            ApplyHeightFromRows(selectionRect);

            var sentenceAnchorMin = sentenceRect.anchorMin;
            sentenceAnchorMin.y = 0.5f;
            sentenceRect.anchorMin = sentenceAnchorMin;

            var sentenceAnchorMax = sentenceRect.anchorMax;
            sentenceAnchorMax.y = 0.5f;
            sentenceRect.anchorMax = sentenceAnchorMax;

            var sentencePivot = sentenceRect.pivot;
            sentencePivot.y = 0.5f;
            sentenceRect.pivot = sentencePivot;

            var selectionAnchorMin = selectionRect.anchorMin;
            selectionAnchorMin.y = 0.5f;
            selectionRect.anchorMin = selectionAnchorMin;

            var selectionAnchorMax = selectionRect.anchorMax;
            selectionAnchorMax.y = 0.5f;
            selectionRect.anchorMax = selectionAnchorMax;

            var selectionPivot = selectionRect.pivot;
            selectionPivot.y = 0.5f;
            selectionRect.pivot = selectionPivot;

            var sentenceHeight = sentenceRect.rect.height > 0f ? sentenceRect.rect.height : sentenceRect.sizeDelta.y;
            var selectionHeight = selectionRect.rect.height > 0f ? selectionRect.rect.height : selectionRect.sizeDelta.y;
            var halfSpacing = sentenceSelectionSpacing * 0.5f;

            var sentencePosition = sentenceRect.anchoredPosition;
            sentencePosition.y = halfSpacing + (selectionHeight * 0.5f);
            sentenceRect.anchoredPosition = sentencePosition;

            var selectionPosition = selectionRect.anchoredPosition;
            selectionPosition.y = -(halfSpacing + (sentenceHeight * 0.5f));
            selectionRect.anchoredPosition = selectionPosition;
        }

        private static int CountActiveChildren(RectTransform parent)
        {
            var count = 0;
            for (int i = 0; i < parent.childCount; i++)
            {
                var child = parent.GetChild(i);
                if (child != null && child.gameObject.activeSelf)
                {
                    count++;
                }
            }

            return count;
        }

        private void ApplyHeightFromRows(RectTransform areaRect)
        {
            var requiredHeight = CalculateRequiredHeightForRows(areaRect);
            var size = areaRect.sizeDelta;
            size.y = requiredHeight;
            areaRect.sizeDelta = size;
        }

        private float CalculateRequiredHeightForRows(RectTransform areaRect)
        {
            if (areaRect == null)
            {
                return 0f;
            }

            var layoutGroup = areaRect.GetComponent<LayoutGroup>();
            var paddingTop = 0f;
            var paddingBottom = 0f;
            var paddingHorizontal = 0f;
            var horizontalSpacing = 0f;
            var verticalSpacing = 0f;

            if (layoutGroup != null)
            {
                paddingTop = layoutGroup.padding.top;
                paddingBottom = layoutGroup.padding.bottom;
                paddingHorizontal = layoutGroup.padding.left + layoutGroup.padding.right;

                var horizontalOrVertical = layoutGroup as HorizontalOrVerticalLayoutGroup;
                if (horizontalOrVertical != null)
                {
                    horizontalSpacing = horizontalOrVertical.spacing;
                    verticalSpacing = horizontalOrVertical.spacing;
                }
            }

            var availableWidth = Mathf.Max(0f, areaRect.rect.width - paddingHorizontal);
            if (availableWidth <= 0f)
            {
                return paddingTop + paddingBottom;
            }

            var totalRowsHeight = 0f;
            var currentRowWidth = 0f;
            var currentRowHeight = 0f;
            var rowCount = 0;

            for (int i = 0; i < areaRect.childCount; i++)
            {
                var child = areaRect.GetChild(i) as RectTransform;
                if (child == null || !child.gameObject.activeSelf)
                {
                    continue;
                }

                var childWidth = ResolveChildSize(child, 0);
                var childHeight = ResolveChildSize(child, 1);

                if (rowCount == 0)
                {
                    rowCount = 1;
                    currentRowWidth = childWidth;
                    currentRowHeight = childHeight;
                    continue;
                }

                var rowWidthWithChild = currentRowWidth + horizontalSpacing + childWidth;
                if (rowWidthWithChild > availableWidth)
                {
                    totalRowsHeight += currentRowHeight;
                    rowCount++;
                    currentRowWidth = childWidth;
                    currentRowHeight = childHeight;
                }
                else
                {
                    currentRowWidth = rowWidthWithChild;
                    currentRowHeight = Mathf.Max(currentRowHeight, childHeight);
                }
            }

            if (rowCount > 0)
            {
                totalRowsHeight += currentRowHeight;
            }

            var totalVerticalSpacing = rowCount > 1 ? (rowCount - 1) * verticalSpacing : 0f;
            return paddingTop + totalRowsHeight + totalVerticalSpacing + paddingBottom;
        }

        private static float ResolveChildSize(RectTransform child, int axis)
        {
            var preferred = LayoutUtility.GetPreferredSize(child, axis);
            if (preferred > 0f)
            {
                return preferred;
            }

            var minimum = LayoutUtility.GetMinSize(child, axis);
            if (minimum > 0f)
            {
                return minimum;
            }

            return axis == 0 ? child.rect.width : child.rect.height;
        }
    }
}
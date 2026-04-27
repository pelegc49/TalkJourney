using System;
using System.Collections.Generic;
using UnityEngine;
using TalkJourney.BubbleSystem.Bubbles;
using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Events;
using TalkJourney.BubbleSystem.Speech;

namespace TalkJourney.BubbleSystem.Flow
{
    [DisallowMultipleComponent]
    public class StageController : MonoBehaviour, IStageController
    {
        [Header("Stage")]
        public StageData initialStage;

        [Tooltip("If true, initialStage is loaded automatically on Start. Disable to start BubbleSystem manually via StartInitialStage().")]
        public bool autoStartInitialStage = true;

        [Header("Sentence")]
        public SentenceBubbleController sentenceBubbleController;

        [Header("Selection")]
        [Tooltip("Parent transform where selection bubbles are instantiated.")]
        public Transform selectionBubbleParent;

        [Tooltip("Prefab used to instantiate each selection bubble.")]
        public SelectionBubbleController selectionBubblePrefab;

        [Header("Speech")]
        public SelectionSpeechMatcher selectionSpeechMatcher;

        [Header("Dependency Injection")]
        [Tooltip("Component implementing localization service; injected into spawned selection bubble display controllers.")]
        public MonoBehaviour localizationServiceBehaviour;

        [Tooltip("Component implementing audio playback manager; injected into spawned selection bubble display controllers.")]
        public MonoBehaviour audioPlaybackManagerBehaviour;

        [Header("Stage Transition")]
        [Tooltip("Optional animator used to fade stages out and in.")]
        public Animator transitionAnimator;

        [Tooltip("Animator bool parameter used to start the fade-out animation.")]
        public string fadeOutBool = "FadeOut";

        [Tooltip("Animator bool parameter used to start the fade-in animation.")]
        public string fadeInBool = "FadeIn";

        [Tooltip("Fallback duration for the stage fade animation.")]
        public float fadeDuration = 0.5f;

        public StageData CurrentStage { get; private set; }
        public event Action<StageData> StageChanged;

        private readonly List<SelectionBubbleController> _activeSelectionBubbles = new List<SelectionBubbleController>();
        private Coroutine _stageTransitionCoroutine;

        private void Start()
        {
            if (autoStartInitialStage && initialStage != null)
            {
                LoadStage(initialStage);
            }
        }

        /// <summary>
        /// Starts BubbleSystem by loading initialStage. Useful for wiring to a UI button OnClick event.
        /// </summary>
        public void StartInitialStage()
        {
            if (initialStage == null)
            {
                Debug.LogWarning("StageController.StartInitialStage was called but initialStage is not assigned.", this);
                return;
            }

            LoadStage(initialStage);
        }

        public bool TransitionToStage(StageData nextStage)
        {
            return StartStageTransition(nextStage);
        }

        public bool TryHandleSelection(SelectionBubbleData selectionData)
        {
            if (selectionData == null)
            {
                return false;
            }

            if (!HasValidSingleCorrectSelection(CurrentStage))
            {
                Debug.LogError("Current stage has invalid selection setup. Exactly one correct selection is required.", this);
                return false;
            }

            if (!selectionData.isCorrect)
            {
                if (selectionData.nextStage == null)
                {
                    Debug.Log("<color=blue>Selection incorrect (no next stage). Trigger incorrect haptic feedback here.</color>", this);
                }
                else
                {
                    Debug.Log("<color=blue>Selection incorrect (next stage ignored because isCorrect is false). Trigger incorrect haptic feedback here.</color>", this);
                }

                BubbleEventBus.PublishSelectionIncorrect(selectionData);
                return false;
            }

            if (selectionData.nextStage != null)
            {
                Debug.Log("<color=green>Selection correct (has next stage). Trigger correct haptic feedback here.</color>", this);
            }
            else
            {
                Debug.Log("<color=green>Selection correct (terminal). Trigger correct haptic feedback here.</color>", this);
            }

            BubbleEventBus.PublishSelectionCorrect(selectionData);

            if (selectionData.nextStage != null)
            {
                return StartStageTransition(selectionData.nextStage);
            }

            // Terminal rule: correct selection with no next stage completes the journey.
            // Keep current visuals visible so BubbleSystemBootstrap can fade them out,
            // then clear visuals after fade completion.
            BubbleEventBus.PublishJourneyCompleted(CurrentStage, selectionData);
            return true;
        }

        /// <summary>
        /// Clears current stage visuals and active selection tracking.
        /// Intended to run after terminal fade-out completes.
        /// </summary>
        public void ClearCurrentStageVisuals()
        {
            ClearSelectionBubbles();
            PushActiveSelectionsToSpeechMatcher();

            if (sentenceBubbleController != null)
            {
                sentenceBubbleController.ClearStageVisuals();
            }
        }

        public bool LoadStage(StageData nextStage)
        {
            if (nextStage == null)
            {
                Debug.LogWarning("Cannot load null stage.", this);
                return false;
            }

            if (!HasValidSingleCorrectSelection(nextStage))
            {
                Debug.LogError($"Stage '{nextStage.name}' must contain exactly one correct selection bubble.", this);
                return false;
            }

            CurrentStage = nextStage;

            if (sentenceBubbleController != null)
            {
                sentenceBubbleController.LoadStage(nextStage);
            }

            RebuildSelectionBubbles(nextStage);
            StageChanged?.Invoke(nextStage);
            BubbleEventBus.PublishStageChanged(nextStage);
            return true;
        }

        private bool StartStageTransition(StageData nextStage)
        {
            if (nextStage == null)
            {
                return false;
            }

            if (_stageTransitionCoroutine != null)
            {
                return false;
            }

            if (transitionAnimator == null)
            {
                return LoadStage(nextStage);
            }

            _stageTransitionCoroutine = StartCoroutine(StageTransitionRoutine(nextStage));
            return true;
        }

        private System.Collections.IEnumerator StageTransitionRoutine(StageData nextStage)
        {
            if (transitionAnimator != null && !string.IsNullOrEmpty(fadeOutBool))
            {
                transitionAnimator.SetBool(fadeOutBool, true);
            }

            yield return new WaitForSeconds(Mathf.Max(0f, fadeDuration));

            if (transitionAnimator != null && !string.IsNullOrEmpty(fadeOutBool))
            {
                transitionAnimator.SetBool(fadeOutBool, false);
            }

            LoadStage(nextStage);

            if (transitionAnimator != null && !string.IsNullOrEmpty(fadeInBool))
            {
                transitionAnimator.SetBool(fadeInBool, true);
            }

            yield return new WaitForSeconds(Mathf.Max(0f, fadeDuration));

            if (transitionAnimator != null && !string.IsNullOrEmpty(fadeInBool))
            {
                transitionAnimator.SetBool(fadeInBool, false);
            }

            _stageTransitionCoroutine = null;
        }

        private bool HasValidSingleCorrectSelection(StageData stageData)
        {
            if (stageData == null || stageData.selectionBubbles == null || stageData.selectionBubbles.Count == 0)
            {
                return false;
            }

            var correctCount = 0;
            for (int i = 0; i < stageData.selectionBubbles.Count; i++)
            {
                var selection = stageData.selectionBubbles[i];
                if (selection != null && selection.isCorrect)
                {
                    correctCount++;
                    if (correctCount > 1)
                    {
                        return false;
                    }
                }
            }

            return correctCount == 1;
        }

        private void RebuildSelectionBubbles(StageData stageData)
        {
            ClearSelectionBubbles();

            if (selectionBubbleParent == null || selectionBubblePrefab == null || stageData == null)
            {
                PushActiveSelectionsToSpeechMatcher();
                return;
            }

            for (int i = 0; i < stageData.selectionBubbles.Count; i++)
            {
                var selectionData = stageData.selectionBubbles[i];
                if (selectionData == null)
                {
                    continue;
                }

                var instance = Instantiate(selectionBubblePrefab, selectionBubbleParent);
                instance.selectionData = selectionData;
                instance.stageControllerBehaviour = this;
                instance.localizationServiceBehaviour = localizationServiceBehaviour;
                instance.RefreshDependencies();
                _activeSelectionBubbles.Add(instance);

                var displayBubble = instance.GetComponent<DisplayBubbleController>();
                if (displayBubble != null)
                {
                    displayBubble.matchPrimaryTextAndDisplayBubbleToContent = true;
                    displayBubble.bubbleData = selectionData.bubble;
                    displayBubble.localizationServiceBehaviour = localizationServiceBehaviour;
                    displayBubble.audioPlaybackManagerBehaviour = audioPlaybackManagerBehaviour;
                    displayBubble.RefreshDependencies();
                    displayBubble.RefreshLocalizedTexts();
                }
            }

            PushActiveSelectionsToSpeechMatcher();
        }

        private void PushActiveSelectionsToSpeechMatcher()
        {
            if (selectionSpeechMatcher != null)
            {
                selectionSpeechMatcher.SetActiveSelectionBubbles(_activeSelectionBubbles);
            }
        }

        private void ClearSelectionBubbles()
        {
            for (int i = 0; i < _activeSelectionBubbles.Count; i++)
            {
                var bubble = _activeSelectionBubbles[i];
                if (bubble != null)
                {
                    Destroy(bubble.gameObject);
                }
            }

            _activeSelectionBubbles.Clear();
        }
    }
}

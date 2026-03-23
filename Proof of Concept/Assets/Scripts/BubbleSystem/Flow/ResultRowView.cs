using UnityEngine;

namespace TalkJourney.BubbleSystem.Flow
{
    /// <summary>
    /// Defines the container slots used by JourneyResultsPresenter when spawning a label/value row.
    /// </summary>
    [DisallowMultipleComponent]
    public class ResultRowView : MonoBehaviour
    {
        [Tooltip("Container where the metric label bubble is spawned.")]
        public Transform labelBubbleParent;

        [Tooltip("Container where the metric value bubble is spawned.")]
        public Transform valueBubbleParent;

        public bool HasSeparateSlots => labelBubbleParent != null && valueBubbleParent != null;

        public Transform ResolveLabelParent()
        {
            return labelBubbleParent != null ? labelBubbleParent : transform;
        }

        public Transform ResolveValueParent()
        {
            return valueBubbleParent != null ? valueBubbleParent : transform;
        }
    }
}
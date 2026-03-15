using System;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.EventSystems;

namespace TalkJourney.BubbleSystem.Interaction
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Collider))]
    public class VrPointerInteractable : MonoBehaviour, IVrPointerInteractable, IPointerEnterHandler, IPointerExitHandler, IPointerClickHandler
    {
        public event Action HoverEntered;
        public event Action HoverExited;
        public event Action Clicked;

        [Header("Optional Unity Events")]
        public UnityEvent onHoverEntered;
        public UnityEvent onHoverExited;
        public UnityEvent onClicked;

        public void OnPointerEnter(PointerEventData eventData)
        {
            HoverEntered?.Invoke();
            onHoverEntered?.Invoke();
        }

        public void OnPointerExit(PointerEventData eventData)
        {
            HoverExited?.Invoke();
            onHoverExited?.Invoke();
        }

        public void OnPointerClick(PointerEventData eventData)
        {
            Clicked?.Invoke();
            onClicked?.Invoke();
        }

        public void InvokeHoverEntered()
        {
            HoverEntered?.Invoke();
            onHoverEntered?.Invoke();
        }

        public void InvokeHoverExited()
        {
            HoverExited?.Invoke();
            onHoverExited?.Invoke();
        }

        public void InvokeClicked()
        {
            Clicked?.Invoke();
            onClicked?.Invoke();
        }
    }
}

using System;

namespace TalkJourney.BubbleSystem.Interaction
{
    public interface IVrPointerInteractable
    {
        event Action HoverEntered;
        event Action HoverExited;
        event Action Clicked;
    }
}

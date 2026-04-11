using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactables;
using UnityEngine.XR.Interaction.Toolkit.Interactors;

public class MinimapVisibility : MonoBehaviour
{
    private XRGrabInteractable grabInteractable;
    private MeshRenderer meshRenderer;

    void Awake()
    {
        grabInteractable = GetComponent<XRGrabInteractable>();
        meshRenderer = GetComponent<MeshRenderer>();
    }

    void OnEnable()
    {
        grabInteractable.selectEntered.AddListener(OnGrabbed);
    }

    void OnDisable()
    {
        grabInteractable.selectEntered.RemoveListener(OnGrabbed);
    }

    private void OnGrabbed(SelectEnterEventArgs args)
    {
        // If pulled by anything other than a socket, turn the mesh back on
        if (!(args.interactorObject is XRSocketInteractor))
        {
            meshRenderer.enabled = true;
            foreach (Transform child in transform)
            {
                // Turn off the GameObject attached to that child Transform
                child.gameObject.SetActive(true);
            }
        }

    }

    // The socket calls this when the tablet snaps into the chest
    public void HideMap()
    {
        if (meshRenderer != null)
        {
            meshRenderer.enabled = false;
            foreach (Transform child in transform)
            {
                // Turn off the GameObject attached to that child Transform
                child.gameObject.SetActive(false);
            }
        }
    }
}
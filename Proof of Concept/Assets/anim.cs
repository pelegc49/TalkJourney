using UnityEngine;

public class anim : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Debug.Log("Script Started");
    }

    // Update is called once per frame
    void Update()
    {
        transform.position =new Vector3(3* Mathf.Sin(2*Time.time), transform.position.y, transform.position.z);

    }
}

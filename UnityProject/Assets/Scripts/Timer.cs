using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Timer : MonoBehaviour
{
    public GameManager gameManager;
    public Slider slider;

    private float startTime;

    // Start is called before the first frame update
    void Start()
    {
        startTime = Time.time;
    }

    // Update is called once per frame
    void Update()
    {
        float timeElapsed = Time.time - startTime;
        
        slider.value = slider.maxValue - timeElapsed;
        if (slider.value <= 0)
            gameManager.LaunchVideo();
    }

    private void OnEnable()
    {
        startTime = Time.time;
        slider.value = slider.maxValue;
    }
}

﻿using EmteqLabs.Models;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.UI;
using TMPro;


namespace EmteqLabs
{
    public class HeartRateDemo : MonoBehaviour
    {
        [SerializeField] private GameObject _currentHRInstructions;

        [SerializeField] private TMP_Text _statusText;
        [SerializeField] private TMP_Text _currentHRText;

        [SerializeField] private GameObject _baselineInstructions;
        [SerializeField] private GameObject _baselinePanels;
        [SerializeField] private Button _calculateBaselineButton;
        [FormerlySerializedAs("_showBaselineButton")] [SerializeField] private Button _showResultsButton;
        [SerializeField] private TMP_Text _standardDeviationHRText;
        [SerializeField] private TMP_Text _medianHRText;

        private void Start()
        {
            // Sync framerate to monitors refresh rate
            QualitySettings.vSyncCount = 1;

            if (!EmteqVRManager.IsDeviceConnected())
            {
                EmteqVRManager.OnDeviceConnect += OnEmteqDeviceConnectionSuccess;
                EmteqVRManager.OnDeviceDisconnect += OnEmteqDeviceConnectionError;
                _statusText.text = ("<color=#00FFFF>Connecting to EmteqVR Device</color>");
            }
            else
            {
                OnEmteqDeviceConnectionSuccess();
            }
        }

        private void OnEmteqDeviceConnectionError()
        {
            _statusText.text = ("<color=#FF4500>Could not connect to EmteqVR Device</color>");
        }

        private void OnEmteqDeviceConnectionSuccess()
        {
            _statusText.text = ("<color=#00FFFF>Detecting Heart Rate...</color>");
            EmteqVRManager.OnHeartRateAverageUpdate += OnHeartRateUpdate;
        }

        private void OnHeartRateUpdate(double hr)
        {
            if (hr > 0d)
            {
                _statusText.text = ("<color=#00FF7F>Heart Rate Detected...</color>");
                _currentHRInstructions.SetActive(false);
                _currentHRText.text = hr.ToString("F");
            }
        }

        //called from button in Unity
        public void CalculateBaseline()
        {
            _calculateBaselineButton.gameObject.SetActive(false);
            _showResultsButton.gameObject.SetActive(true);
            _baselineInstructions.SetActive(false);
            _baselinePanels.SetActive(true);
            EmteqVRManager.StartHeartRateBaselineCalibration();
        }

        //called from a button in Unity
        public void ShowBaselineResult()
        {
            _showResultsButton.gameObject.SetActive(false);
            BaselineHeartRateData baselineHeartRateData = EmteqVRManager.EndHeartRateBaselineCalibration();
            _standardDeviationHRText.text = baselineHeartRateData.StandardDeviation.ToString("F");
            _medianHRText.text = baselineHeartRateData.Median.ToString("F");
        }

        private void OnDestroy()
        {
            EmteqVRManager.OnHeartRateAverageUpdate -= OnHeartRateUpdate;

            EmteqVRManager.OnDeviceConnect -= OnEmteqDeviceConnectionSuccess;
            EmteqVRManager.OnDeviceDisconnect -= OnEmteqDeviceConnectionError;
        }
    }
}

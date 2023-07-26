using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using EmteqLabs;
using EmteqVR;
using EmteqLabs.Faceplate;
using System;
using System.IO;
using TMPro;

// Data saved in : [PathToApplication]/Results
public class RecordRawEmteqV2 : MonoBehaviour
{
    private const string folder = "Results/";
    private const string emgFilename = "_emg_v2.csv";
    private const string ppgFilename = "_ppg_v2.csv";
    private int participantId = 0;
    private int videoId = 0;

    private bool isRecording = false;

    public List<TMP_Text> sensorLabel;

    private List<double> emgTimestamps = new List<double>();
    private List<double> ppgTimestamps = new List<double>();
    private List<ushort[]> emgData = new List<ushort[]>();
    private List<PpgRawSignal> ppgData = new List<PpgRawSignal>();

    // Update is called once per frame
    void Update()
    {
        if (isRecording)
        {
            ushort[] newEmgData = new ushort[sensorLabel.Count];
            emgTimestamps.Add(DateTime.UtcNow.Subtract(new DateTime(1970, 1, 1)).TotalSeconds);
            for (int i = 0; i < sensorLabel.Count; i++)
            {
                newEmgData[i] = Convert.ToUInt16(sensorLabel[i].text);
            }
            emgData.Add(newEmgData);

            Dictionary<MuscleMapping, ushort> currentEMG;
            PpgRawSignal currentPPG;

            GetData(out currentEMG, out currentPPG);

            ppgData.Add(currentPPG);
            ppgTimestamps.Add(DateTime.UtcNow.Subtract(new DateTime(1970, 1, 1)).TotalSeconds);
        }
    }

    public void StartRecording()
    {
        isRecording = true;
    }

    public void StopRecording()
    {
        isRecording = false;
    }

    private void GetData(out Dictionary<MuscleMapping, ushort> currentEMG, out PpgRawSignal currentPPG)
    {
        currentEMG = EmteqVRManager.GetEmgAmplitudeRms();

        currentPPG = EmteqVRManager.GetRawPpgSignal();
    }

    public void SaveData()
    {
        // Write EMG data csv file
        Directory.CreateDirectory(folder);
        string emgPath = folder + participantId.ToString() + "_" + videoId.ToString() + emgFilename;
        using (StreamWriter sw = new StreamWriter(emgPath))
        {
            sw.WriteLine("timestamp;" +
                "LeftZygomaticus;RightZygomaticus;" +
                "LeftOrbicularis;RightOrbicularis;" +
                "LeftFrontalis;RightFrontalis;" +
                "CenterCorrugator;");
            for (int i = 0; i < emgData.Count; i++)
            {
                string line = emgTimestamps[i].ToString() + ";" +
                    emgData[i][0].ToString() + ";" + emgData[i][1].ToString() + ";" +
                    emgData[i][2].ToString() + ";" + emgData[i][3].ToString() + ";" +
                    emgData[i][4].ToString() + ";" + emgData[i][5].ToString() + ";" +
                    emgData[i][6].ToString() + ";";
                line = line.Replace(",", ".");
                sw.WriteLine(line);
            }
        }

        // Write PPG data csv file
        string ppgPath = folder + participantId.ToString() + "_" + videoId.ToString() + ppgFilename;
        using (StreamWriter sw = new StreamWriter(ppgPath))
        {
            sw.WriteLine("timestamp;" +
                "Ppg;Proximity;");
            for (int i = 0; i < ppgData.Count; i++)
            {
                string line = ppgTimestamps[i].ToString() + ";" +
                    ppgData[i].Ppg.ToString() + ";" + ppgData[i].Proximity.ToString() + ";";
                line = line.Replace(",", ".");
                sw.WriteLine(line);
            }
        }
    }

    public void SetParticpantId(int id)
    {
        participantId = id;
    }

    public void SetVideoId(int id)
    {
        videoId = id;
    }

    public void ClearData()
    {
        emgTimestamps.Clear();
        emgData.Clear();
        ppgTimestamps.Clear();
        ppgData.Clear();
    }
}

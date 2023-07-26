using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

public class RecordManager : MonoBehaviour
{
    // True : Part1 ; False : Part2
    public bool part1;

    public RecordFaceTrackerData recordFaceTrackerData;
    public RecordEmteqData recordEmteqData;
    public RecordRawEmteq recordRawEmteq;
    public RecordRawEmteqV2 recordRawEmteqV2;

    public Slider arousalSlider;
    public Slider valenceSlider;

    private int participantId;
    private List<string[]> ratingData = new List<string[]>();

    // Start is called before the first frame update
    void Start()
    {
        string[] tmp = new string[3];
        tmp[0] = "VideoID";
        tmp[1] = "Arousal";
        tmp[2] = "Valence";
        ratingData.Add(tmp);
    }

    public void SetParticipantId(int id)
    {
        participantId = id;
        recordFaceTrackerData.SetParticpantId(id);
        recordRawEmteq.SetParticpantId(id);
        recordRawEmteqV2.SetParticpantId(id);
        recordEmteqData.SetParticipantId(id);
    }

    public void StartRecording(int videoId)
    {
        if (part1)
        {
            recordEmteqData.StartSection(videoId);
            recordFaceTrackerData.SetVideoId(videoId);
            recordRawEmteq.SetVideoId(videoId);
            recordRawEmteqV2.SetVideoId(videoId);
        }
        else
        {
            recordEmteqData.StartSection(videoId+7);
            recordFaceTrackerData.SetVideoId(videoId+7);
            recordRawEmteq.SetVideoId(videoId + 7);
            recordRawEmteqV2.SetVideoId(videoId + 7);
        }
        recordFaceTrackerData.StartRecording();
        recordRawEmteq.StartRecording();
        recordRawEmteqV2.StartRecording();
    }

    public void StopRecording(int videoId)
    {
        if(part1)
            recordEmteqData.EndSection(videoId);
        else
            recordEmteqData.EndSection(videoId+7);
        recordFaceTrackerData.StopRecording();
        recordRawEmteq.StopRecording();
        recordRawEmteqV2.StopRecording();
        recordFaceTrackerData.SaveData();
        recordRawEmteq.SaveData();
        recordRawEmteqV2.SaveData();
        recordFaceTrackerData.ClearData();
        recordRawEmteq.ClearData();
        recordRawEmteqV2.ClearData();
    }

    public void AddLastRating(int videoId)
    {
        string[] tmp = new string[3];
        if(part1)
            tmp[0] = videoId.ToString();
        else    
            tmp[0] = (videoId+7).ToString();
        tmp[1] = arousalSlider.value.ToString().Replace(",", ".");
        tmp[2] = valenceSlider.value.ToString().Replace(",", ".");
        ratingData.Add(tmp);
    }

    public void SaveRating()
    {
        // Write rating csv file
        string filePath = "./Results/" + participantId.ToString() + "_rating.csv";
        string delimiter = ";";

        StringBuilder sb = new StringBuilder();

        foreach (string[] row in ratingData)
        {
            sb.AppendLine(string.Join(delimiter, row));
        }

        StreamWriter outStream = System.IO.File.CreateText(filePath);
        outStream.WriteLine(sb);
        outStream.Close();
    }
}

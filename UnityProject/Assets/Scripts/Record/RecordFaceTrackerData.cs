using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;
using VIVE;
using VIVE.FacialTracking;

// Data saved in : [PathToApplication]/Results
public class RecordFaceTrackerData : MonoBehaviour
{
	private const string folder = "Results/";
	private const string timestampsFilename = "_EventStarted.csv";
	private const string eyeFilename = "_eye_expression.csv";
	private const string lipFilename = "_mouth_shape.csv";
	private int participantId = 0;
	private int videoId = 0;

	private FacialManager eyemanager = new FacialManager();
	private FacialManager lipmanager = new FacialManager();

	private bool isRecording = false;

	private List<double> timestamps = new List<double>();
	private List<double> eyeTimestamp = new List<double>();
	private List<float[]> eyeData = new List<float[]>();
	private List<double> lipTimestamp = new List<double>();
	private List<float[]> lipData = new List<float[]>();

	// Start is called before the first frame update
	void Start()
	{
		eyemanager.StartFramework(XrFacialTrackingTypeHTC.XR_FACIAL_TRACKING_TYPE_EYE_DEFAULT_HTC);

		lipmanager.StartFramework(XrFacialTrackingTypeHTC.XR_FACIAL_TRACKING_TYPE_LIP_DEFAULT_HTC);
	}

	// Update is called once per frame
	void Update()
	{
		if (isRecording)
		{
			Dictionary<XrEyeShapeHTC, float> eyeWeightings;
			Dictionary<XrLipShapeHTC, float> lipWeightings;

			GetData(out eyeWeightings, out lipWeightings);

			float[] newEyeData = new float[eyeWeightings.Count];
			eyeTimestamp.Add(DateTime.UtcNow.Subtract(new DateTime(1970, 1, 1)).TotalSeconds);
			for (int i = 0; i < eyeWeightings.Count; i++)
			{
				newEyeData[i] = eyeWeightings.ElementAt(i).Value;
			}
			eyeData.Add(newEyeData);

			float[] newLipData = new float[lipWeightings.Count];
			lipTimestamp.Add(DateTime.UtcNow.Subtract(new DateTime(1970, 1, 1)).TotalSeconds);
			for (int i = 0; i < lipWeightings.Count; i++)
			{
				newLipData[i] = lipWeightings.ElementAt(i).Value;
			}
			lipData.Add(newLipData);
		}
	}

	public void StartRecording()
    {
		isRecording = true;
		timestamps.Add(DateTime.UtcNow.Subtract(new DateTime(1970, 1, 1)).TotalSeconds);
	}

	public void StopRecording()
    {
		isRecording = false;
        timestamps.Add(DateTime.UtcNow.Subtract(new DateTime(1970, 1, 1)).TotalSeconds);
	}

	// Get eye tracking and lip tracking data
	private void GetData(out Dictionary<XrEyeShapeHTC, float> eyeWeightings, out Dictionary<XrLipShapeHTC, float> lipWeightings)
	{
		eyemanager.GetWeightings(out eyeWeightings);

		lipmanager.GetWeightings(out lipWeightings);
	}

	public void SaveData()
	{
		// Write timestamps csv file
		Directory.CreateDirectory(folder);
		string timestampsPath = folder + participantId.ToString() + "_" + videoId.ToString() + timestampsFilename;
		using (StreamWriter sw = new StreamWriter(timestampsPath))
		{
			sw.WriteLine("timestamp;string;");
			for (int i = 0; i < timestamps.Count; i++)
			{
				string line = timestamps[i].ToString() + ";";
				if ((i % 2) == 0)
					line += "Démarrage chrono;";
				else
					line += "Fin chrono;";
				line = line.Replace(",", ".");
				sw.WriteLine(line);
			}

		}

		// Write eye tracking data csv file
		string eyePath = folder + participantId.ToString() + "_" + videoId.ToString() + eyeFilename;
		using (StreamWriter sw = new StreamWriter(eyePath))
		{
			sw.WriteLine("timestamp;" +
				"LeftWide;LeftSqueeze;LeftFrown;" +
				"RightWide;RightSqueeze;RightFrown;");
			for (int i = 0; i < eyeData.Count; i++)
			{
				string line = eyeTimestamp[i].ToString() + ";" +
					eyeData[i][1].ToString() + ";" + eyeData[i][4].ToString() + ";" + 0.ToString() + ";" +
					eyeData[i][3].ToString() + ";" + eyeData[i][5].ToString() + ";" + 0.ToString() + ";";
				line = line.Replace(",", ".");
				sw.WriteLine(line);
			}
		}

		// Write lip tracking data csv file
		string lipPath = folder + participantId.ToString() + "_" + videoId.ToString() + lipFilename;
		using (StreamWriter sw = new StreamWriter(lipPath))
		{
			sw.WriteLine("timestamp;" +
				"Mouth ape shape (0-1 key);Mouth upper right (0-1 key);Mouth upper left (0-1 key);Mouth lower right (0-1 key);Mouth lower left ((0-1 key);" +
				"Mouth upper overturn (0-1 key);Mouth lower overturn (0-1 key);Mouth pout (0-1 key);Mouth smile right (0-1 key);Mouth smile left (0-1 key);" +
				"Mouth sad right (0-1 key);Mouth sad left (0-1 key);Mouth upper up right (0-1 key);Mouth upper up left (0-1 key);Mouth lower down right (0-1 key);" +
				"Mouth lower down left (0-1 key);Mouth upper inside (0-1 key);Mouth lower inside (0-1 key);Mouth lower overlay (0-1 key);Jaw right (0-1 key);" +
				"Jaw forward (0-1 key);Cheek puff right (0-1 key);Cheek puff left (0-1 key);Cheek suck (0-1 key);Tong long step 1 (0-1 key);" +
				"Tong long step 2 (0-1 key);Tong down (0-1 key);Tong up (0-1 key);Tong right (0-1 key);Tong left (0-1 key);" +
				"Tong roll (0-1 key);Tong up left morph (0-1 key);Tong up right morph (0-1 key);Tong down left morph (0-1 key);Tong down right morph (0-1 key);");

			for (int i = 0; i < lipData.Count; i++)
			{
				string line = lipTimestamp[i].ToString() + ";"
					+ lipData[i][4] + ";" + lipData[i][5] + ";" + lipData[i][6] + ";" + lipData[i][7] + ";" + lipData[i][8] + ";"
					+ lipData[i][9] + ";" + lipData[i][10] + ";" + lipData[i][11] + ";" + lipData[i][12] + ";" + lipData[i][13] + ";"
					+ lipData[i][14] + ";" + lipData[i][15] + ";" + lipData[i][19] + ";" + lipData[i][20] + ";" + lipData[i][21] + ";"
					+ lipData[i][22] + ";" + lipData[i][23] + ";" + lipData[i][24] + ";" + lipData[i][25] + ";" + lipData[i][0] + ";"
					+ lipData[i][2] + ";" + lipData[i][16] + ";" + lipData[i][17] + ";" + lipData[i][18] + ";" + lipData[i][26] + ";"
					+ lipData[i][32] + ";" + lipData[i][30] + ";" + lipData[i][29] + ";" + lipData[i][28] + ";" + lipData[i][27] + ";"
					+ lipData[i][31] + ";" + lipData[i][34] + ";" + lipData[i][33] + ";" + lipData[i][36] + ";" + lipData[i][35] + ";";
				line = line.Replace(",", ".");
				sw.WriteLine(line);
			}
		}
	}

	private void OnDestroy()
	{
		eyemanager.StopFramework(XrFacialTrackingTypeHTC.XR_FACIAL_TRACKING_TYPE_EYE_DEFAULT_HTC);

		lipmanager.StopFramework(XrFacialTrackingTypeHTC.XR_FACIAL_TRACKING_TYPE_LIP_DEFAULT_HTC);
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
		timestamps.Clear();
		eyeTimestamp.Clear();
		eyeData.Clear();
		lipTimestamp.Clear();
		lipData.Clear();
	}
}

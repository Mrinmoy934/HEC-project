# Android/Flutter API Integration Example

## 1. API Endpoint
**URL**: `POST /alert` or `GET /status` (hypothetical)
**Response Format (JSON)**:

```json
{
  "status": "success",
  "data": {
    "timestamp": "2023-10-27T10:30:00Z",
    "location": "Zone A - Waterhole",
    "risk_level": "High",
    "details": {
      "behaviour": "Aggressive",
      "posture": "Charging",
      "elephant_count": 1
    },
    "alert_message": "DANGER: High Risk Elephant Activity Detected! Evacuate immediately."
  }
}
```

## 2. Flutter Code Example

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';

class ElephantAlertScreen extends StatefulWidget {
  @override
  _ElephantAlertScreenState createState() => _ElephantAlertScreenState();
}

class _ElephantAlertScreenState extends State<ElephantAlertScreen> {
  String riskLevel = "Loading...";
  String message = "";
  Color statusColor = Colors.grey;

  Future<void> fetchRiskStatus() async {
    final response = await http.get(Uri.parse('http://YOUR_SERVER_IP:8000/status'));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body)['data'];
      setState(() {
        riskLevel = data['risk_level'];
        message = data['alert_message'];
        
        if (riskLevel == 'High') {
          statusColor = Colors.red;
        } else if (riskLevel == 'Medium') {
          statusColor = Colors.orange;
        } else {
          statusColor = Colors.green;
        }
      });
    } else {
      setState(() {
        riskLevel = "Error";
      });
    }
  }

  @override
  void initState() {
    super.initState();
    fetchRiskStatus();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Elephant Conflict Alert")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              "Current Risk Level:",
              style: TextStyle(fontSize: 20),
            ),
            SizedBox(height: 20),
            Container(
              padding: EdgeInsets.all(20),
              color: statusColor,
              child: Text(
                riskLevel,
                style: TextStyle(fontSize: 40, color: Colors.white, fontWeight: FontWeight.bold),
              ),
            ),
            SizedBox(height: 20),
            Padding(
              padding: EdgeInsets.all(16.0),
              child: Text(
                message,
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 16),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
```

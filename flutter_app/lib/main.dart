import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  runApp(const PaddyDiseaseApp());
}

class PaddyDiseaseApp extends StatelessWidget {
  const PaddyDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Paddy Doctor AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0F4C3A), // Paddy Emerald
          primary: const Color(0xFF0F4C3A),
          secondary: const Color(0xFF107C69),
          surface: const Color(0xFFF8FAF9),
        ),
        scaffoldBackgroundColor: const Color(0xFFF4F7F5),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF0F4C3A),
          foregroundColor: Colors.white,
          elevation: 2,
          centerTitle: true,
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  Uint8List? _selectedImageBytes;
  String? _selectedImageName;
  bool _isLoading = false;
  String _serverUrl = "http://192.168.0.100:8000"; // Default server IP
  Map<String, dynamic>? _resultData;
  String _loadingStatus = "Analyzing Paddy Leaf...";

  @override
  void initState() {
    super.initState();
    _loadSavedServerUrl();
  }

  Future<void> _loadSavedServerUrl() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _serverUrl = prefs.getString("server_url") ?? "http://192.168.0.100:8000";
    });
  }

  Future<void> _saveServerUrl(String newUrl) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString("server_url", newUrl);
    setState(() {
      _serverUrl = newUrl;
    });
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1080,
        maxHeight: 1080,
        imageQuality: 85,
      );

      if (pickedFile != null) {
        final bytes = await pickedFile.readAsBytes();
        setState(() {
          _selectedImageBytes = bytes;
          _selectedImageName = pickedFile.name;
          _resultData = null;
        });
        _sendImageForDiagnosis();
      }
    } catch (e) {
      _showErrorDialog("Failed to access camera or gallery: $e");
    }
  }

  Future<void> _sendImageForDiagnosis() async {
    if (_selectedImageBytes == null) return;

    setState(() {
      _isLoading = true;
      _loadingStatus = "1/4. Running EfficientNet-B0 Classifier...";
    });

    try {
      final uri = Uri.parse("$_serverUrl/predict");
      var request = http.MultipartRequest("POST", uri);

      request.files.add(
        http.MultipartFile.fromBytes(
          "file",
          _selectedImageBytes!,
          filename: _selectedImageName ?? "leaf.jpg",
        ),
      );

      // Simulate step progression
      Future.delayed(const Duration(milliseconds: 600), () {
        if (mounted && _isLoading) {
          setState(() {
            _loadingStatus = "2/4. Generating Grad-CAM++ Attribution...";
          });
        }
      });
      Future.delayed(const Duration(milliseconds: 1400), () {
        if (mounted && _isLoading) {
          setState(() {
            _loadingStatus = "3/4. Segmenting with Prompted SAM...";
          });
        }
      });
      Future.delayed(const Duration(milliseconds: 2200), () {
        if (mounted && _isLoading) {
          setState(() {
            _loadingStatus = "4/4. Quantifying Canopy Severity (DIP)...";
          });
        }
      });

      var streamedResponse = await request.send().timeout(const Duration(seconds: 45));
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final decoded = json.decode(response.body);
        setState(() {
          _resultData = decoded;
          _isLoading = false;
        });
      } else {
        _showErrorDialog("Server Error (${response.statusCode}):\n${response.body}");
        setState(() => _isLoading = false);
      }
    } catch (e) {
      _showErrorDialog(
        "Could not connect to server at:\n$_serverUrl\n\n"
        "Checklist:\n"
        "1. Is the laptop running 'python server.py'?\n"
        "2. Is your phone connected to the same Wi-Fi / Laptop Hotspot?\n"
        "3. Did you enter the correct Laptop IP in settings?",
      );
      setState(() => _isLoading = false);
    }
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.error_outline, color: Colors.red),
            SizedBox(width: 8),
            Text("Connection Error"),
          ],
        ),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text("OK"),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(ctx);
              _showServerConfigDialog();
            },
            child: const Text("Change IP"),
          ),
        ],
      ),
    );
  }

  void _showServerConfigDialog() {
    final controller = TextEditingController(text: _serverUrl);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("Configure Laptop Server IP"),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Enter the IP of your laptop running 'server.py'.\n"
              "Find it by typing 'ipconfig' in your laptop terminal.",
              style: TextStyle(fontSize: 12, color: Colors.black54),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: controller,
              decoration: const InputDecoration(
                labelText: "Server URL",
                hintText: "http://192.168.1.15:8000",
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.wifi),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text("Cancel"),
          ),
          ElevatedButton(
            onPressed: () {
              _saveServerUrl(controller.text.trim());
              Navigator.pop(ctx);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text("Server URL updated to: ${controller.text.trim()}")),
              );
            },
            child: const Text("Save"),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.eco, color: Color(0xFF86EFAC)),
            SizedBox(width: 8),
            Text("Paddy Doctor AI", style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: "Server Settings",
            onPressed: _showServerConfigDialog,
          ),
        ],
      ),
      body: _isLoading
          ? _buildLoadingState()
          : _resultData != null
              ? _buildResultsDashboard()
              : _buildCaptureWelcomeState(),
    );
  }

  Widget _buildCaptureWelcomeState() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          // Banner Card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [Color(0xFF0F4C3A), Color(0xFF166534)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(16),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.12),
                  blurRadius: 10,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF59E0B),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: const Text(
                    "EfficientNet-B0 + SAM ViT-B",
                    style: TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold),
                  ),
                ),
                const SizedBox(height: 12),
                const Text(
                  "Paddy Disease Diagnostic & Severity System",
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 6),
                const Text(
                  "Instant 10-Class Disease Recognition, Zero-Shot Lesion Segmentation & Pathological Severity %",
                  style: TextStyle(color: Color(0xFFD1FAE5), fontSize: 13),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // Selected Image Preview (if chosen)
          if (_selectedImageBytes != null)
            ClipRRect(
              borderRadius: BorderRadius.circular(14),
              child: Image.memory(_selectedImageBytes!, height: 220, width: double.infinity, fit: BoxFit.cover),
            )
          else
            Container(
              height: 220,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.grey.shade300, width: 2),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.add_a_photo_outlined, size: 64, color: Colors.grey.shade400),
                  const SizedBox(height: 12),
                  Text(
                    "Capture a Clear Paddy Leaf Photo",
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.grey.shade700),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    "Hold phone 15-20 cm from the leaf surface",
                    style: TextStyle(fontSize: 12, color: Colors.grey.shade500),
                  ),
                ],
              ),
            ),

          const SizedBox(height: 28),

          // Primary Capture Buttons
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt, color: Colors.white),
                  label: const Text(
                    "Take Photo",
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF0F4C3A),
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library, color: Color(0xFF0F4C3A)),
                  label: const Text(
                    "Gallery",
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Color(0xFF0F4C3A)),
                  ),
                  style: OutlinedButton.styleFrom(
                    side: const BorderSide(color: Color(0xFF0F4C3A), width: 1.8),
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 24),
          // Server Connection Indicator
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Colors.grey.shade300),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.wifi_tethering, size: 16, color: Colors.green),
                const SizedBox(width: 6),
                Text(
                  "Backend: $_serverUrl",
                  style: const TextStyle(fontSize: 11, color: Colors.black87),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLoadingState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(
              color: Color(0xFF0F4C3A),
              strokeWidth: 4,
            ),
            const SizedBox(height: 28),
            Text(
              _loadingStatus,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Color(0xFF0F4C3A)),
            ),
            const SizedBox(height: 10),
            const Text(
              "Zero-Shot Segmentation & Higher-Order Spatial Attribution",
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultsDashboard() {
    final pred = _resultData!["prediction"];
    final vis = _resultData!["visualizations"];

    final isNormal = pred["class_name"] == "normal";
    final severityPct = (pred["severity_percentage"] as num).toDouble();
    final severityTier = pred["severity_category"] as String;
    final confPct = pred["confidence_percentage"];

    Color severityColor = Colors.green;
    if (severityPct > 50) {
      severityColor = Colors.red;
    } else if (severityPct > 25) {
      severityColor = Colors.deepOrange;
    } else if (severityPct > 10) {
      severityColor = Colors.orange;
    } else if (!isNormal) {
      severityColor = Colors.amber.shade700;
    }

    return DefaultTabController(
      length: 3,
      child: Scaffold(
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // 1. Disease Diagnosis Card
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(color: Colors.black.withOpacity(0.06), blurRadius: 10, offset: const Offset(0, 3)),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: isNormal ? Colors.green.shade100 : Colors.red.shade100,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            isNormal ? "HEALTHY CROP" : "DISEASE DETECTED",
                            style: TextStyle(
                              color: isNormal ? Colors.green.shade800 : Colors.red.shade800,
                              fontWeight: FontWeight.bold,
                              fontSize: 11,
                            ),
                          ),
                        ),
                        Text(
                          "Confidence: $confPct%",
                          style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.black54, fontSize: 13),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(
                      pred["display_name"],
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: isNormal ? const Color(0xFF166534) : const Color(0xFF991B1B),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 16),

              // 2. Severity Meter Card
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(color: Colors.black.withOpacity(0.06), blurRadius: 10, offset: const Offset(0, 3)),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          "Infection Severity:",
                          style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.black87),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: severityColor.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            severityTier,
                            style: TextStyle(color: severityColor, fontWeight: FontWeight.bold, fontSize: 12),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Text(
                          "${severityPct.toStringAsFixed(1)}%",
                          style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: severityColor),
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(8),
                            child: LinearProgressIndicator(
                              value: severityPct / 100.0,
                              minHeight: 12,
                              backgroundColor: Colors.grey.shade200,
                              valueColor: AlwaysStoppedAnimation<Color>(severityColor),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(
                      "Segmented Lesion: ${pred['lesion_pixels']} px  |  Canopy Area: ${pred['leaf_pixels']} px",
                      style: const TextStyle(fontSize: 11, color: Colors.black54),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 16),

              // 3. Multi-Tab Visualizations (Original, SAM Overlay, Grad-CAM++)
              Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(color: Colors.black.withOpacity(0.06), blurRadius: 10, offset: const Offset(0, 3)),
                  ],
                ),
                child: Column(
                  children: [
                    const TabBar(
                      labelColor: Color(0xFF0F4C3A),
                      unselectedLabelColor: Colors.grey,
                      indicatorColor: Color(0xFF0F4C3A),
                      indicatorWeight: 3,
                      tabs: [
                        Tab(text: "SAM Overlay"),
                        Tab(text: "Grad-CAM++"),
                        Tab(text: "Original"),
                      ],
                    ),
                    SizedBox(
                      height: 240,
                      child: TabBarView(
                        children: [
                          _buildBase64Image(vis["composite_overlay_base64"]),
                          _buildBase64Image(vis["gradcam_heatmap_base64"]),
                          Image.memory(_selectedImageBytes!, fit: BoxFit.contain),
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 16),

              // 4. Agronomic Prescription & Treatment Card
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: const Color(0xFFEFF6FF), // Soft Blue
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: const Color(0xFFBFDBFE)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Row(
                      children: [
                        Icon(Icons.medical_services_outlined, color: Color(0xFF1D4ED8)),
                        SizedBox(width: 8),
                        Text(
                          "Recommended Field Action:",
                          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15, color: Color(0xFF1E3A8A)),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      pred["treatment_recommendation"],
                      style: const TextStyle(fontSize: 13, color: Color(0xFF1E293B), height: 1.4),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // 5. Scan Another Button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () {
                    setState(() {
                      _selectedImageBytes = null;
                      _resultData = null;
                    });
                  },
                  icon: const Icon(Icons.refresh, color: Colors.white),
                  label: const Text("Scan Another Leaf", style: TextStyle(color: Colors.white, fontSize: 16)),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF0F4C3A),
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                ),
              ),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildBase64Image(String base64Str) {
    Uint8List bytes = base64.decode(base64Str);
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: Image.memory(bytes, fit: BoxFit.contain),
      ),
    );
  }
}


import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const StudyMateApp());
}

class StudyMateApp extends StatelessWidget {
  const StudyMateApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'StudyMate AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const StudyMateHome(),
    );
  }
}

class StudyMateHome extends StatefulWidget {
  const StudyMateHome({super.key});

  @override
  State<StudyMateHome> createState() => _StudyMateHomeState();
}

class _StudyMateHomeState extends State<StudyMateHome> {
  late final WebViewController controller;
  bool isLoading = true;
  bool hasError = false;

  @override
  void initState() {
    super.initState();

    controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(Colors.white)
      ..setNavigationDelegate(
        NavigationDelegate(
          onPageStarted: (_) {
            setState(() {
              isLoading = true;
              hasError = false;
            });
          },
          onPageFinished: (_) {
            setState(() => isLoading = false);
          },
          onWebResourceError: (error) {
            if (error.isForMainFrame == true) {
              setState(() {
                isLoading = false;
                hasError = true;
              });
            }
          },
        ),
      )
      ..loadRequest(
        Uri.parse(
          'https://studymate-ai-jpozmaggylelgaxgqpz6bj.streamlit.app/',
        ),
      );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🎓 StudyMate AI'),
        centerTitle: true,
        actions: [
          IconButton(
            tooltip: 'Refresh',
            icon: const Icon(Icons.refresh),
            onPressed: () => controller.reload(),
          ),
        ],
      ),
      body: Stack(
        children: [
          WebViewWidget(controller: controller),
          if (isLoading)
            const Center(child: CircularProgressIndicator()),
          if (hasError)
            Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text('StudyMate AI load nahi ho pa raha.'),
                  const SizedBox(height: 12),
                  ElevatedButton(
                    onPressed: () => controller.reload(),
                    child: const Text('Try Again'),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
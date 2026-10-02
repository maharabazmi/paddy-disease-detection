import 'package:flutter_test/flutter_test.dart';
import 'package:paddy_disease_detector/main.dart';

void main() {
  testWidgets('Paddy Doctor AI smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const PaddyDiseaseApp());
    expect(find.text('Paddy Doctor AI'), findsWidgets);
  });
}

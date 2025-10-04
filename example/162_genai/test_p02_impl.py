
import unittest
from unittest.mock import patch

# Import the code to be tested and the mock interface
import p02_impl
from p02_impl import GenAIJob, GenerationConfig
from google.genai import types  # Import the real 'types' for comparing enum values
from mock_genai import MockGenAI


class TestP02ImplPlayback(unittest.TestCase):
    """
    Test suite for p02_impl.GenAIJob using a mock playback interface.
    The expected results are hardcoded for easy maintenance. If the model
    output changes, a developer can copy the new output from the logs and
    paste it directly into the variables below.
    """

    def test_genai_job_run_with_mock_playback(self):
        """
        Tests the GenAIJob.run method by patching the genai interface.
        The mock plays back data from the YAML file, and the results are
        compared with hardcoded ground-truth values.
        """
        # --- Expected Values: Copy and paste from the log file here ---

        expected_thoughts = """**Initiating Company Research**

I'm starting by compiling a list of comparable European industrial giants to Bosch and Siemens. The goal is to identify core business areas for each firm. Once I have the preliminary list, I'll dive into their individual areas of focus to determine the most relevant "topics" for each one. The output of this stage will be a list of companies that match the criteria, along with a high level overview of the areas where they operate.


**Defining Scope & Criteria**

I've refined my approach by breaking down the company analysis into specific steps. I'll begin by identifying the major European industrial and technology firms comparable to Bosch and Siemens. Then, I will delve into their core business areas, key innovation strategies, and sources of competitive advantage. The output will be a detailed, topic-grouped list.


**Expanding the Scope**

I've just broadened the scope to look at key European tech and industrial giants, starting with a base of Bosch and Siemens. This initial sweep included direct rivals like ABB and Schneider Electric, along with software players like Dassault Systèmes and SAP.  Now, I'm ready to delve deeper.


**Compiling Key Data**

I'm now collating the core data points for the initial list of companies. The goal is a concise summary of their primary innovations and competitive advantages. I've selected ABB, Schneider, ASML, Dassault, Continental, and BASF for detailed analysis. I'm focusing on distilling "innovation" and "moat" info into a useful, structured form, going beyond the initial broad strokes. The data is piling up, but the structure is forming.


**Analyzing European Leaders**

I've just added to my initial data collection, focusing specifically on ABB and Schneider Electric, delving into their innovations and competitive moats within the industrial tech and automation realm. The next step is ASML and Dassault Systèmes, researching their advanced technology and software offerings to further refine the list.


**Exploring Industrial Leaders**

I've just finished compiling detailed information on ABB and Schneider Electric, both prominent in the Industrial Technology & Automation field. ABB's strengths lie in industrial automation, robotics, and electrification, with a strong emphasis on digital solutions and their ABB Ability software.


**Reviewing Industry Titans**

I've deepened my dive into ABB and Schneider Electric, focusing on their innovation. I've highlighted ABB's digitalization, robotics, and power efficiency, particularly their ABB Ability software. With Schneider Electric, I've honed in on EcoStruxure, AI, and their strong sustainability focus, giving them a core competitive advantage.


**Shifting Focus, Broadening Scope**

I'm now diving into ASML, Dassault Systèmes, Continental, and BASF, expanding my scope beyond industrial automation. I'm prioritizing innovation and competitive advantages, just as before, ensuring I understand their core strengths. I'm aiming to categorize them effectively and succinctly. The research will feed into a more comprehensive final report, still aiming for topic grouping and in-depth entries.


**Evaluating Recent Data**

I've just successfully gathered in-depth data on the remaining companies. My focus now is on ASML's advancements, particularly their Extreme Ultraviolet (EUV) lithography and High-NA EUV systems, which are crucial for 2nm chips and beyond. I'm also digging into computational lithography as it relates to this.


**Examining New Developments**

I've just successfully researched and analyzed recent data on Dassault Systèmes and Continental. I'm focusing on the 3DEXPERIENCE platform and its use of generative AI. I am also investigating the impact on competition of Dassault's dominance in 3D design software. My attention has turned to Continental's advancements in automotive technology, especially in autonomous driving and electrification, as these are promising.


**Deep Diving into Moats**

I've just finished a deep dive into the moats of ASML, Dassault Systèmes, Continental, and BASF. ASML's near-monopoly and advanced tech are formidable. Dassault's high switching costs and network effect are key. Continental's resilient tires business and strong R&D are noteworthy. Finally, BASF's Verbund system and global leadership are a significant advantage.


**Summarizing Final Data**

I've wrapped up compiling detailed information on all selected companies, focusing on ASML's EUV lithography and its role in 2nm chip development, Dassault's 3DEXPERIENCE platform with generative AI, Continental's advancements in autonomous driving and electrification, and BASF's sustainable chemistry initiatives and Verbund system. My next step is to format all the research."""

        expected_answer = """The European industrial landscape is characterized by a group of highly specialized and diversified conglomerates similar to Bosch and Siemens. These companies maintain their market leadership through deep R&D investment, a focus on the Industrial Internet of Things (IIoT), sustainability, and decades-long customer relationships, forming powerful economic "moats."

Here is a list of major European industrial and technology companies, grouped by their core focus, innovation drivers, and competitive moat.

| Company | Country | Topic (Core Business) | Key Innovation Focus | Competitive Moat (Durable Advantage) |
| :--- | :--- | :--- | :--- | :--- |
| **ABB** | Switzerland / Sweden | Industrial Automation, Robotics, Electrification & Motion | **Digitalization & Sustainable Solutions:** Integrated digital-industrial platform (*ABB Ability™*), advanced AI-driven robotics (e.g., collaborative robots, AMRs), and high-efficiency power products for smart grids. | **Proprietary Technology & Installed Base:** Extensive intellectual property in robotics and power systems, decades of reliability building trust with large industrial clients, and a vast global service network. |
| **Schneider Electric** | France | Energy Management, Digital Automation, Smart Infrastructure | **Energy Digitalization:** IoT-enabled *EcoStruxure™* platform for optimizing energy in buildings, data centers, and industries, and pioneering Energy-as-a-Service (EaaS) business models. | **Sustainability as a Competitive Edge:** Market leadership in a mission-critical field (energy efficiency) backed by significant customer switching costs from integrated software/hardware solutions and a focus on decarbonization. |
| **Dassault Systèmes** | France | 3D Design Software, Product Lifecycle Management (PLM), Virtual Twin | **Virtual Twin Technology & AI:** The **3DEXPERIENCE platform** for creating virtual replicas of products and processes, integrating Generative AI, and defining "3D UNIV+RSES" (virtual-plus-real environments). | **High Switching Costs & Network Effect:** Deep entrenchment in critical industries (e.g., used to design 90% of all aircraft), creating high switching costs for major customers and a *de facto* industry standard. |
| **ASML** | Netherlands | Semiconductor Manufacturing Equipment (Photolithography) | **Extreme Ultraviolet (EUV) Lithography:** Sole supplier of the world's most advanced lithography machines, pioneering High-NA EUV for producing next-generation microchips (2nm nodes and below). | **Technological Near-Monopoly:** A unique and indispensable position as the *only* provider of EUV systems, backed by decades of R&D, massive IP, and immense capital barriers to entry for competitors. |
| **BASF** | Germany | Chemicals, Advanced Materials, Agricultural Solutions | **Sustainable Chemistry & Digital R&D:** Developing products with reduced or net-zero Product Carbon Footprint (PCF), materials for battery electric vehicle (BEV) components, and accelerating R&D using digitalization and a proprietary supercomputer (*Quriosity*). | **"Verbund" Production System:** A highly integrated, large-scale production setup that creates cost efficiencies and resource optimization at major sites that competitors struggle to replicate, maintaining a global top-three market position in core businesses. |
| **Continental AG** | Germany | Automotive Technology, Tires, Industrial Solutions | **Future Mobility:** Heavy investment in advanced driver-assistance systems (ADAS), software-defined vehicle architectures, electrification of powertrains, and digital/smart tire solutions. | **Diversification & Legacy Ties:** The combination of a highly profitable and resilient **Tires business** that generates significant cash flow, coupled with long-standing, deep-seated supply relationships with global automotive manufacturers (OEMs). |
| **Airbus** | Netherlands / France | Aerospace and Defence | **Sustainable Aviation:** Developing zero-emission flights, pioneering hydrogen-powered aircraft concepts, and continuous innovation in aircraft materials for better fuel efficiency. | **Regulatory & Programmatic Moat:** Extremely high regulatory barriers (certification process for new aircraft), colossal initial R&D costs, and a duopoly structure (with Boeing) creating a near-guaranteed market share for large-scale commercial programs. |"""

        expected_usage = {
            'candidates_token_count': 1007,
            'prompt_token_count': 19,
            'thoughts_token_count': 1740,
            'response_id': 'FMLgaIHWJJTrnsEPr9WSsQ4',
            'model_version': 'gemini-2.5-flash-preview-09-2025',
            'total_token_count': 4707,
            'finish_reason': types.FinishReason.STOP,
        }

        # --- Test Execution ---

        yaml_playback_file = 'european_companies_error_out_20251004_06_43_24.yaml'
        mock_genai_interface = MockGenAI(yaml_playback_file=yaml_playback_file)

        with patch('p02_impl.genai', mock_genai_interface):
            # 1. Configure the job.
            config = GenerationConfig(
                prompt_text="This prompt will be ignored by the mock.",
                output_yaml_path="test_playback_output.yaml"
            )

            # 2. Instantiate and run the job.
            job = GenAIJob(config)
            result = job.run()

            # 3. Assert that results match the expected log output.
            # Using .strip() makes the comparison robust against leading/trailing whitespace.
            self.assertEqual(result.thoughts.strip(), expected_thoughts.strip())
            self.assertEqual(result.answer.strip(), expected_answer.strip())

            # 4. Assert that the usage summary matches the expected values.
            # Asserting key by key provides clearer error messages on failure.
            for key, value in expected_usage.items():
                self.assertEqual(result.usage_summary[key], value, f"Usage key '{key}' does not match.")


if __name__ == '__main__':
    unittest.main()

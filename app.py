import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
#PARENT_PATH = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

from src.preprocess import preprocess_raw_data

import joblib

# Data loading with cache
@st.cache_data
def load_data() -> pd.DataFrame:
	# Cached so it doesn't run on every interaction
	df = preprocess_raw_data(save_interim = False)
	return df

def compute_rates(df: pd.DataFrame) -> dict:
	# This will compute overall engagement rates
	if df is None or df.empty:
		return{
			"open_rate": np.nan,
			"click_rate": np.nan,
			"conversion_rate": np.nan,
			"unsubscribe_rate": np.nan,
			"n_rows": 0,
		}
	
	return{
		"open_rate": df["open_flag"].mean() if "open_flag" in df.columns else np.nan,
		"click_rate": df["click_flag"].mean() if "click_flag" in df.columns else np.nan,
		"conversion_rate": df["conversion_flag"].mean() if "conversion_flag" in df.columns else np.nan,
		"unsubscribe_rate": df["unsubscribe_flag"].mean() if "unsubscribe_flag" in df.columns else np.nan,
		"n_rows": len(df),
	}

@st.cache_resource
def get_model_and_preprocessor():
	model = joblib.load(ARTIFACT_DIR / "rf_model_deploy.joblib")
	preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
	metadata = joblib.load(ARTIFACT_DIR / "metadata.joblib")
	raw_feature_cols = metadata["raw_feature_cols"]
	return model, preprocessor, metadata, raw_feature_cols

def multiselect_with_select_all(label:str, options, state_key:str):
	# Sidebar expander
	# Select all checkbox
	# Multiselect options

	# If there are no options, a broken widget is not be rendered
	if not options:
		return[]
	
	all_key = f"{state_key}_all"
	multi_key = f"{state_key}_values"

	# Initialization
	if all_key not in st.session_state:
		st.session_state[all_key] = True

	if multi_key not in st.session_state:
		st.session_state[multi_key] = list(options)

	def _update_from_select_all():
		# when the checkbox is toggled, set or keep values
		if st.session_state[all_key]:
			st.session_state[multi_key] = list(options) # selecting everything
	
	def _update_from_multiselect():
		# This would sync the "Select All" box when multiselect changes
		selected = st.session_state[multi_key]
		st.session_state[all_key] = (len(selected) == len(options))
	
	with st.sidebar.expander(label):
		st.checkbox(
			"Select all", key=all_key, on_change=_update_from_select_all,
		)
		st.multiselect(
			label, options=options, key=multi_key,
			on_change = _update_from_multiselect
		)
	
	return st.session_state[multi_key]


def main():
	st.set_page_config(
		page_title = "Email Marketing Campaign Analysis",
		page_icon="📧",
		layout="wide",
		#width="stretch"
	)
	
	st.title("📧Email Marketing Campaign Analytics")
	st.write("""
		This dashboard is built on a synthetic dataset (~400000rows) for an email marketing scenario.
		You can explore engagement patterns by segment, device, time of day and more.
	""")
	
	model, preprocessor, metadata, raw_feature_cols = get_model_and_preprocessor()

	dropdowns = metadata.get("dropdowns", {})
	country_options = dropdowns.get("country", [])
	device_type_options = dropdowns.get("device_type", [])
	archetype_options = dropdowns.get("consumer_archetypes", [])
	mosaic_options = dropdowns.get("mosaic_segment", [])
	mailing_category_options = dropdowns.get("mailing_category", [])

	st.sidebar.markdown("### Data Loading")
	load_full = st.sidebar.toggle("Load full dataset (~400k rows)", value=False)
	
	df = None
	if load_full:
		with st.spinner("Loadidng and preprocessign full dataset..."):
			df = load_data()
	else:
		st.info("App is running in lightweight mode. Turn on 'Load full dataset' in sidebar for full exploration.")
	
	dropdowns = metadata.get("dropdowns", {})

	# Creating Tabs
	tab_overview, tab_segments, tab_score, tab_shap = st.tabs(
		["Overview", "Segmentation Explorer", "Open Probability Predictor", "SHAP"]
	)

	# ----------------------------------Tab 1 -> Overview
	with tab_overview:
		st.subheader("Overall Engagement Summary")

		if df is None:
			st.warning("Overview needs the dataset. Turn on 'Load full dataset' in the sidebar.")
			st.stop()

		base_metrics = compute_rates(df)

		c1, c2, c3, c4 = st.columns(4)
		c1.metric("Open Rate", f"{base_metrics['open_rate']*100:.1f}%")
		c2.metric("Click Rate", f"{base_metrics['click_rate']*100:.1f}%")
		c3.metric("Conversion Rate", f"{base_metrics['conversion_rate']*100:.2f}%")
		c4.metric("Unsubscribe Rate", f"{base_metrics['unsubscribe_rate']*100:.2f}%")

		st.caption((f"Total rows: {base_metrics['n_rows']:,}"))

		col_left, col_right = st.columns(2)

		# Open rate by hour
		if "mailing_hour" in df.columns and "open_flag" in df.columns:
			with col_left:
				st.markdown("#### Open Rate by Hour of Day")
				hour_view = (
					df.groupby("mailing_hour", observed=False)["open_flag"]
					.mean().reset_index().sort_values("mailing_hour")
				)
				st.line_chart(
					hour_view.set_index("mailing_hour"),
					width="stretch",
				)
		
		# Open rate by device
		if "device_type" in df.columns and "open_flag" in df.columns:
			with col_right:
				st.markdown("#### Open Rate by Device")
				device_view = (
					df.groupby("device_type", observed=False)["open_flag"]
					.mean().sort_values(ascending=False).reset_index()
				)
				st.bar_chart(
					device_view.set_index("device_type")["open_flag"],
					width="stretch",
				)
	
	# ----------------------------------Tab 2 -> Segmentation Explorer
	with tab_segments:
		st.subheader("Segmentation Explorer")
		st.write("Use the filters in the sidebar to slice the dataset and see how engagement changes.")

		st.sidebar.markdown("### Filters (Segmentation Explorer)")

		countries = country_options
		archetypes = archetype_options
		mosaic_segments = mosaic_options
		devices = device_type_options
		categories = mailing_category_options

		selected_countries = multiselect_with_select_all(
			label="Country", options = countries, state_key="country"
		)

		selected_archetypes = multiselect_with_select_all(
			label = "Consumer Archetypes", options = archetypes, state_key="consumer_archetypes"
		)

		selected_mosaic = multiselect_with_select_all(
			label = "Mosaic Segment", options  = mosaic_segments, state_key= "mosaic_segment"
		)

		selected_devices = multiselect_with_select_all(
			label ="Device Type", options= devices, state_key = "device_type"
		)

		selected_categories = multiselect_with_select_all(
			label = "Mailing Category", options = categories, state_key = "mailing_category"
		)

		if df is None:
			st.warning("Segmentation Explorer needs the dataset. Tun on 'Load full dataset' in the sidebar.")
			st.stop()
			
		# Applying filters
		filtered = df.copy()
		if selected_countries:
			filtered = filtered[filtered["country"].isin(selected_countries)]
		if selected_archetypes:
			filtered = filtered[filtered["consumer_archetypes"].isin(selected_archetypes)]
		if selected_mosaic:
			filtered = filtered[filtered["mosaic_segment"].isin(selected_mosaic)]
		if selected_devices:
			filtered = filtered[filtered["device_type"].isin(selected_devices)]
		if selected_categories:
			filtered = filtered[filtered["mailing_category"].isin(selected_categories)]
		
		# Metrics for comparison - filtered vs Overall
		base_metrics = compute_rates(df)
		filtered_metrics = compute_rates(filtered)
		
		col1, col2, col3, col4 = st.columns(4)
		col1.metric(
		"Open Rate (filtered)",
		f"{filtered_metrics['open_rate']*100:.1f}%",
		f"{(filtered_metrics['open_rate']-base_metrics['open_rate'])*100:.1f} pts vs overall",
		)
		col2.metric(
			"Click Rate (filtered)",
			f"{filtered_metrics['click_rate']*100:.1f}%",
			f"{(filtered_metrics['click_rate']-base_metrics['click_rate'])*100:.1f} pts vs overall",
		)
		col3.metric(
			"Conversion Rate (filtered)",
			f"{filtered_metrics['conversion_rate']*100:.2f}%",
			f"{(filtered_metrics['conversion_rate']-base_metrics['conversion_rate'])*100:.2f} pts vs overall",
		)
		col4.metric(
			"Unsubscribe Rate (filtered)",
			f"{filtered_metrics['unsubscribe_rate']*100:.2f}%",
			f"{(filtered_metrics['unsubscribe_rate']-base_metrics['unsubscribe_rate'])*100:.2f} pts vs overall",
		)
		
		st.caption(
			f"Filtered rows: {filtered_metrics['n_rows']:,} / Total rows: {base_metrics['n_rows']:,}"
		)
		
		# Layout : Two columns with charts
		left_col, right_col = st.columns(2)
		
		# Open rate by hour
		if "mailing_hour" in filtered.columns and "open_flag" in filtered.columns:
			with left_col:
				st.subheader("Open Rate by Hour of Day")
				hour_view = (
					filtered.groupby("mailing_hour", observed=False)["open_flag"]
					.mean().reset_index().sort_values("mailing_hour")
				)
				st.line_chart(
					hour_view.set_index("mailing_hour"),
					width="stretch",
					)
		
		# Open rate by consumer archetype
		if "consumer_archetypes" in filtered.columns and "open_flag" in filtered.columns:
			with right_col:
				st.subheader("Open Rate by Consumer Archetype")
				arch_view = (
					filtered.groupby("consumer_archetypes", observed=False)["open_flag"]
					.mean().sort_values(ascending=False).reset_index()
				)
				st.bar_chart(
					arch_view.set_index("consumer_archetypes")["open_flag"],
					width="stretch",
				)
		
		# Second row : Device + Mosaic
		left2, right2 = st.columns(2)
		
		if "device_type" in filtered.columns and "click_flag" in filtered.columns:
			with left2:
				st.subheader("Click Rate by Device Type")
				dev_view = (
					filtered.groupby("device_type", observed=False)["click_flag"]
					.mean().sort_values(ascending=False).reset_index()
				)
				st.bar_chart(
					dev_view.set_index("device_type")["click_flag"],
					width="stretch",
					)
		if "mosaic_segment" in filtered.columns and "open_flag" in filtered.columns:
			with right2:
				st.subheader("Open Rate by Mosaic Segment")
				mosaic_view = (
					filtered.groupby("mosaic_segment", observed=False)["open_flag"]
					.mean().sort_values(ascending=False).reset_index()
				)
				st.bar_chart(
					mosaic_view.set_index("mosaic_segment")["open_flag"],
					width="stretch",
					)	
		
		# Sample Data to be displayed
		st.markdown("#### Sample of Filtered Data (Key Columns)")
		cols_to_show= [
			c for c in [
				"country", "region", "consumer_archetypes", "mosaic_segment", "device_type",
				"mailing_category", "mailing_hour", "open_flag", "click_flag", "conversion_flag",
			]
			if c in filtered.columns
		]
		st.dataframe(
			filtered[cols_to_show].head(50),
			width="stretch",
			height=320,
			)
	
	# ----------------------------------Tab 3 -> Open Probability Predictor
	with tab_score:
		if df is None:
			st.warning("Predictor needs the dataset. Turn on 'Load full dataset' in the sidebar.")
			st.stop()
		st.subheader("Open Probability Predictor")

		st.write(
			"""
			This section will allow you to enter customer attributes and get a predicted open probability
			from the trained model.
			"""
		)

		with st.form("scoring_form"):
			col_a, col_b, col_c = st.columns(3)

			with col_a:
				age = st.number_input("Age", min_value=18, max_value=90, value=36)
				country = st.selectbox(
					"Country", options=country_options)
			
				device_type = st.selectbox(
					"Device Type", options=device_type_options)
				

			with col_b:
				archetype = st.selectbox(
					"Consumer Archetype", options=archetype_options)

				mosaic_segment = st.selectbox(
					"Mosaic Segment", options=mosaic_options)

				mailing_hour = st.slider("Mailing Hour", 0, 23, 9)
			
			with col_c:
				previous_open = st.slider(
					"Previous Open Rate", min_value =0.0, max_value=1.0, value = 0.3, step = 0.01
				)
				previous_click = st.slider(
					"Previous Click Rate", min_value=0.0, max_value=1.0, value = 0.05, step = 0.01
				)
				previous_purchases = st.number_input(
					"Previous Purchases", min_value = 0, max_value = 50, value = 1
				)

			submitted = st.form_submit_button("Predict Open Probability")
		
		if submitted:
			template = df.iloc[[0]].copy()

			# Override with user inputs
			template["age"] = age
			template["country"] = country
			template["device_type"] = device_type
			template["consumer_archetypes"] = archetype
			template["mosaic_segment"] = mosaic_segment
			template["mailing_hour"] = mailing_hour
			template["previous_open_rate"] = previous_open
			template["previous_click_rate"] = previous_click
			template["previous_purchases"] = previous_purchases

			# Computing engagement_score
			template["engagement_score"] = (
				0.6 * template["previous_open_rate"] +
				0.3 * template["previous_click_rate"] +
				0.1 * (template["previous_purchases"] / (df["previous_purchases"].max() + 1))
			)

			# Transforming
			missing = [c for c in raw_feature_cols if c not in template.columns]
			if missing:
				st.error(f"Template missing columns required by preprocessor: {missing[:10]} ...")
				st.stop()

			X_single = preprocessor.transform(template[raw_feature_cols])

			prob = float(model.predict_proba(X_single)[0, 1])

			st.success(f"Estimated open probability: **{prob * 100:.1f}%**")

			st.markdown("** Prediction logic (input snapshot) **")

			st.json({
				"age": age,
				"country": country,
				"device_type": device_type,
				"consumer_archetypes": archetype,
				"mosaic_segment": mosaic_segment,
				"mailing_hour": mailing_hour,
				"previous_open_rate": previous_open,
				"previous_click_rate": previous_click,
				"previous_purchases": previous_purchases,
			})
	
	# ----------------------------------Tab 4 SHAP
	with tab_shap:
		st.subheader("Model Explainability with SHAP")
		st.write(
			"""
			These plots explain **why** the model predicts higher of lower open probability for different customers.

			- The **bar plot** shows which features are most important overall.
			- The **summary (dot) plot** shows how a feature values impact our predictions.
			- The **dependence plot** shows how a single feature (like 'previous_open_rate') influences the prediction.
			"""
		)

		fig_dir = PROJECT_ROOT / "reports" / "figures"

		col1, col2 = st.columns(2)
	
		# Bar plot
		with col1:
			st.markdown("### Global Importance (Bar)")
			bar_path = fig_dir / "shap_summary_bar.png"
			if bar_path.exists():
				st.image(str(bar_path), width="stretch")
			else:
				st.info("SHAP bar plot not found. Please run the SHAP notebook first.")
		
		# Dot 
		with col2:
			st.markdown("### Global Impact (Summary Plot)")
			dot_path = fig_dir / "shap_summary_dot.png"
			if dot_path.exists():
				st.image(str(dot_path), width="stretch")
			else:
				st.info("SHAP summary plot not found. Please run the SHAP notebook first.")
		
		st.markdown("### Dependence: 'previous_open_rate' vs SHAP value")
		dep_path = fig_dir / "shap_dependencies_previous_open_rate.png"
		if dep_path.exists():
			st.image(str(dep_path), width="stretch")
		else:
			st.info("SHAP dependence plot not found. Please run the SHAP notebook first.")

		st.markdown("#### Local Explanation Example (Force Plot)")
		force_path = fig_dir / "shap_force_example.png"
		if force_path.exists():
			st.image(str(force_path), width="stretch")
		else:
			st.info("SHAP force plot not found. Please run the SHAP notebook first.")
	
if __name__ == "__main__":
	main()
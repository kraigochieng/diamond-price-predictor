<script setup lang="ts">
import { ref, onMounted, watch } from "vue";
import type { DiamondInput, DiamondOutput } from "@/types/diamond";

// form state
const carat = ref<number>(0.5);
const result = ref<(DiamondOutput & { createdAt: string }) | null>(null);
const history = ref<(DiamondOutput & { createdAt: string })[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);

// currency state
const currency = ref<"USD" | "EUR" | "KES">("USD");
const rates = ref<{ [key: string]: number }>({ USD: 1 }); // start with USD only

// fetch exchange rates once on mount
const fetchRates = async () => {
	try {
		const res = await $fetch<{ rates: Record<string, number> }>(
			"https://api.frankfurter.dev/v1/latest?base=USD"
		);
		// add USD manually because API doesn’t return it
		rates.value = { USD: 1, ...res.rates };
		console.log(rates.value);
	} catch (e) {
		console.error("Failed to fetch exchange rates:", e);
	}
};
onMounted(fetchRates);

// watch currency changes (ensure it exists)
watch(currency, () => {
	if (!rates.value[currency.value]) {
		fetchRates();
	}
});

const submit = async () => {
	loading.value = true;
	error.value = null;
	result.value = null;

	const payload: DiamondInput = { carat: carat.value };

	const config = useRuntimeConfig();

	try {
		const res = await $fetch<DiamondOutput>(
			`${config.public.apiBase}/predict`,
			{
				method: "POST",
				body: payload,
			}
		);

		// add timestamp
		const withTime = { ...res, createdAt: new Date().toLocaleString() };

		result.value = withTime;
		history.value.unshift(withTime);
	} catch (e: any) {
		error.value = e?.data?.detail || e?.message || "Something went wrong.";
	} finally {
		loading.value = false;
	}
};

// format price with currency conversion
const formatPrice = (value: number) => {
	const rate = rates.value[currency.value] || 1;
	const converted = value * rate;

	return new Intl.NumberFormat("en-US", {
		style: "currency",
		currency: currency.value,
	}).format(converted);
};
</script>
<template>
	<div class="max-w-5xl mx-auto py-10 px-4">
		<div class="flex gap-2 items-center align-items">
			<UIcon name="i-lucide-diamond" class="size-8" />
			<h1 class="text-2xl font-bold">Diamond Price Predictor</h1>
		</div>

		<p class="text-gray-600 mb-6">
			This tool predicts prices for
			<strong>small to medium diamonds</strong>
			(0.2–2 carats).
		</p>

		<!-- Responsive Grid -->
		<div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
			<!-- Left Side: Form -->
			<div>
				<!-- Currency Selector -->
				<div class="mb-6">
					<label class="block text-sm font-medium text-gray-700 mb-1">
						Display Currency
					</label>
					<USelect
						v-model="currency"
						class="w-full"
						:items="[
							{ label: 'USD ($)', value: 'USD' },
							{ label: 'EUR (€)', value: 'EUR' },
							{ label: 'KES (KSh)', value: 'KES' },
						]"
					/>
				</div>

				<!-- Input -->
				<UForm @submit.prevent="submit">
					<UFormField
						:label="`Carat size (0.2 – 2): ${carat.toFixed(2)}`"
						name="carat"
					>
						<div>
							<USlider
								v-model="carat"
								color="neutral"
								class="w-full"
								tooltip
								:min="0.2"
								:max="2"
								:step="0.01"
							/>
							<div
								class="flex justify-between text-sm text-gray-500 mt-1"
							>
								<span>0.2</span>
								<span>2.0</span>
							</div>
							<p class="mt-2">
								Selected: {{ carat.toFixed(2) }} carats
							</p>
						</div>
					</UFormField>

					<UButton
						type="submit"
						:loading="loading"
						icon="i-lucide-rocket"
						color="neutral"
						class="p-4 mt-4 w-full flex items-center justify-center"
					>
						Predict Price
					</UButton>
				</UForm>

				<!-- Error -->
				<div v-if="error" class="text-red-500 mt-4">
					{{ error }}
				</div>
			</div>

			<!-- Right Side: Latest Prediction + Feed -->
			<div class="flex flex-col">
				<!--  -->
				<UCard v-if="result" class="mt-6" variant="subtle">
					<template #header>
						<h2 class="text-xl font-semibold">Latest Prediction</h2>
					</template>

					<div class="space-y-2">
						<p>Message: {{ result.message }}</p>
						<p>
							Predicted Price:
							<strong>{{ formatPrice(result.preds) }}</strong>
						</p>
						<p class="text-sm text-gray-500">
							Input: {{ result.data.carat.toFixed(2) }} carats
						</p>
					</div>

					<template #footer>
						<p class="text-xs text-gray-400">
							{{ result.createdAt }}
						</p>
					</template>
				</UCard>
				<h2 class="text-lg font-semibold mb-3">Prediction Feed</h2>

				<p v-if="history.length === 0" class="text-gray-500">
					No predictions yet. Use the slider and click
					<strong>Predict Price</strong>.
				</p>

				<!-- Scrollable feed -->
				<div
					v-else
					class="space-y-3 overflow-y-auto pr-2"
					style="max-height: 500px"
				>
					<UCard
						v-for="(item, index) in history"
						:key="index"
						variant="subtle"
					>
						<template #header>
							<p class="font-medium">
								Carat: {{ item.data.carat.toFixed(2) }}
							</p>
						</template>

						<p>
							Price:
							<span class="font-medium">
								{{ formatPrice(item.preds) }}
							</span>
						</p>

						<template #footer>
							<p class="text-xs text-gray-400">
								{{ item.createdAt }}
							</p>
						</template>
					</UCard>
				</div>
			</div>
		</div>

		<!-- Attribution -->
		<p class="text-xs text-gray-500 pt-6">
			Currency rates are provided by
			<a
				href="https://www.frankfurter.dev/"
				target="_blank"
				rel="noopener"
				class="underline"
			>
				Frankfurter </a
			>, a free open-source API that tracks reference exchange rates
			published by institutional and non-commercial sources like the
			European Central Bank.
		</p>
	</div>
</template>

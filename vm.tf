resource "google_compute_instance" "instance-20250125-025754" {
  boot_disk {
    auto_delete = true
    device_name = "instance-20250125-025754"

    initialize_params {
      image = "projects/debian-cloud/global/images/debian-11-bullseye-v20250123"
      size  = 500 #TOTAL BOOT DISK SIZE
      type  = "pd-balanced"
    }

    mode = "READ_WRITE"
  }

  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false

  guest_accelerator {
    count = 1
    type  = "projects/tokushukai-gcp06/zones/us-central1-a/acceleratorTypes/nvidia-a100-80gb" #ACCELERATOR (GPU NAME)
  }

  labels = {
    goog-ec-src = "vm_add-tf"
  }

  machine_type = "a2-ultragpu-1g" #SCALE IT UP IF GPU ALSO INCREASES

  metadata = {
    enable-oslogin = "true"
  }

  name = "instance-20250125-025754" #INSTANCE_NAME

  network_interface {
    access_config {
      network_tier = "PREMIUM"
    }

    queue_count = 0
    stack_type  = "IPV4_ONLY"
    subnetwork  = "projects/tokushukai-gcp06/regions/us-central1/subnetworks/default"
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "TERMINATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  service_account {
    email  = "761719187648-compute@developer.gserviceaccount.com"
    scopes = ["https://www.googleapis.com/auth/devstorage.read_only", "https://www.googleapis.com/auth/logging.write", "https://www.googleapis.com/auth/monitoring.write", "https://www.googleapis.com/auth/service.management.readonly", "https://www.googleapis.com/auth/servicecontrol", "https://www.googleapis.com/auth/trace.append"]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  zone = "us-central1-a"
}

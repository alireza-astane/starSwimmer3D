#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.1726280152135813,-0.08751412356284648,-0.49803758662398845>, 1 }        
    sphere {  m*<0.20067188720586826,0.11207230433544443,4.13466401179361>, 1 }
    sphere {  m*<2.562080378792676,0.01451985182352765,-1.7272471120751718>, 1 }
    sphere {  m*<-1.7942433751064715,2.2409598208557524,-1.4719833520399583>, 1}
    sphere { m*<-1.5264561540686397,-2.646732121548145,-1.2824370668773857>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.20067188720586826,0.11207230433544443,4.13466401179361>, <-0.1726280152135813,-0.08751412356284648,-0.49803758662398845>, 0.5 }
    cylinder { m*<2.562080378792676,0.01451985182352765,-1.7272471120751718>, <-0.1726280152135813,-0.08751412356284648,-0.49803758662398845>, 0.5}
    cylinder { m*<-1.7942433751064715,2.2409598208557524,-1.4719833520399583>, <-0.1726280152135813,-0.08751412356284648,-0.49803758662398845>, 0.5 }
    cylinder {  m*<-1.5264561540686397,-2.646732121548145,-1.2824370668773857>, <-0.1726280152135813,-0.08751412356284648,-0.49803758662398845>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.1726280152135813,-0.08751412356284648,-0.49803758662398845>, 1 }        
    sphere {  m*<0.20067188720586826,0.11207230433544443,4.13466401179361>, 1 }
    sphere {  m*<2.562080378792676,0.01451985182352765,-1.7272471120751718>, 1 }
    sphere {  m*<-1.7942433751064715,2.2409598208557524,-1.4719833520399583>, 1}
    sphere { m*<-1.5264561540686397,-2.646732121548145,-1.2824370668773857>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.20067188720586826,0.11207230433544443,4.13466401179361>, <-0.1726280152135813,-0.08751412356284648,-0.49803758662398845>, 0.5 }
    cylinder { m*<2.562080378792676,0.01451985182352765,-1.7272471120751718>, <-0.1726280152135813,-0.08751412356284648,-0.49803758662398845>, 0.5}
    cylinder { m*<-1.7942433751064715,2.2409598208557524,-1.4719833520399583>, <-0.1726280152135813,-0.08751412356284648,-0.49803758662398845>, 0.5 }
    cylinder {  m*<-1.5264561540686397,-2.646732121548145,-1.2824370668773857>, <-0.1726280152135813,-0.08751412356284648,-0.49803758662398845>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
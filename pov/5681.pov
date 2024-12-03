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
    sphere { m*<-1.1471510964764693,-0.170716923777887,-1.2711957490195358>, 1 }        
    sphere {  m*<0.12737142330941587,0.2821854505209542,8.636874908748405>, 1 }
    sphere {  m*<5.781105449180541,0.07216418281866571,-4.789149340571319>, 1 }
    sphere {  m*<-2.8089924763464365,2.1583354095841707,-2.1732981571103602>, 1}
    sphere { m*<-2.541205255308605,-2.7293565328197267,-1.9837518719477898>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12737142330941587,0.2821854505209542,8.636874908748405>, <-1.1471510964764693,-0.170716923777887,-1.2711957490195358>, 0.5 }
    cylinder { m*<5.781105449180541,0.07216418281866571,-4.789149340571319>, <-1.1471510964764693,-0.170716923777887,-1.2711957490195358>, 0.5}
    cylinder { m*<-2.8089924763464365,2.1583354095841707,-2.1732981571103602>, <-1.1471510964764693,-0.170716923777887,-1.2711957490195358>, 0.5 }
    cylinder {  m*<-2.541205255308605,-2.7293565328197267,-1.9837518719477898>, <-1.1471510964764693,-0.170716923777887,-1.2711957490195358>, 0.5}

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
    sphere { m*<-1.1471510964764693,-0.170716923777887,-1.2711957490195358>, 1 }        
    sphere {  m*<0.12737142330941587,0.2821854505209542,8.636874908748405>, 1 }
    sphere {  m*<5.781105449180541,0.07216418281866571,-4.789149340571319>, 1 }
    sphere {  m*<-2.8089924763464365,2.1583354095841707,-2.1732981571103602>, 1}
    sphere { m*<-2.541205255308605,-2.7293565328197267,-1.9837518719477898>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12737142330941587,0.2821854505209542,8.636874908748405>, <-1.1471510964764693,-0.170716923777887,-1.2711957490195358>, 0.5 }
    cylinder { m*<5.781105449180541,0.07216418281866571,-4.789149340571319>, <-1.1471510964764693,-0.170716923777887,-1.2711957490195358>, 0.5}
    cylinder { m*<-2.8089924763464365,2.1583354095841707,-2.1732981571103602>, <-1.1471510964764693,-0.170716923777887,-1.2711957490195358>, 0.5 }
    cylinder {  m*<-2.541205255308605,-2.7293565328197267,-1.9837518719477898>, <-1.1471510964764693,-0.170716923777887,-1.2711957490195358>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
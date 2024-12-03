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
    sphere { m*<1.1381079553753624,0.2445031998189995,0.5387925724546357>, 1 }        
    sphere {  m*<1.382269023540879,0.26316900531346965,3.5287812248305928>, 1 }
    sphere {  m*<3.8755162126034155,0.26316900531346965,-0.6885009836600251>, 1 }
    sphere {  m*<-3.260889324225204,7.3088276429035615,-2.062183468253644>, 1}
    sphere { m*<-3.7552128402591385,-7.982201151851635,-2.3537784299057547>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.382269023540879,0.26316900531346965,3.5287812248305928>, <1.1381079553753624,0.2445031998189995,0.5387925724546357>, 0.5 }
    cylinder { m*<3.8755162126034155,0.26316900531346965,-0.6885009836600251>, <1.1381079553753624,0.2445031998189995,0.5387925724546357>, 0.5}
    cylinder { m*<-3.260889324225204,7.3088276429035615,-2.062183468253644>, <1.1381079553753624,0.2445031998189995,0.5387925724546357>, 0.5 }
    cylinder {  m*<-3.7552128402591385,-7.982201151851635,-2.3537784299057547>, <1.1381079553753624,0.2445031998189995,0.5387925724546357>, 0.5}

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
    sphere { m*<1.1381079553753624,0.2445031998189995,0.5387925724546357>, 1 }        
    sphere {  m*<1.382269023540879,0.26316900531346965,3.5287812248305928>, 1 }
    sphere {  m*<3.8755162126034155,0.26316900531346965,-0.6885009836600251>, 1 }
    sphere {  m*<-3.260889324225204,7.3088276429035615,-2.062183468253644>, 1}
    sphere { m*<-3.7552128402591385,-7.982201151851635,-2.3537784299057547>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.382269023540879,0.26316900531346965,3.5287812248305928>, <1.1381079553753624,0.2445031998189995,0.5387925724546357>, 0.5 }
    cylinder { m*<3.8755162126034155,0.26316900531346965,-0.6885009836600251>, <1.1381079553753624,0.2445031998189995,0.5387925724546357>, 0.5}
    cylinder { m*<-3.260889324225204,7.3088276429035615,-2.062183468253644>, <1.1381079553753624,0.2445031998189995,0.5387925724546357>, 0.5 }
    cylinder {  m*<-3.7552128402591385,-7.982201151851635,-2.3537784299057547>, <1.1381079553753624,0.2445031998189995,0.5387925724546357>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
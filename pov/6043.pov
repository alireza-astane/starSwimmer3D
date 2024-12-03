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
    sphere { m*<-1.5653872507790982,-0.23142109586474635,-1.0123510733590257>, 1 }        
    sphere {  m*<-0.10046969645213388,0.22234276008758008,8.869444322540158>, 1 }
    sphere {  m*<7.254881741547835,0.13342248409322244,-5.710048967505205>, 1 }
    sphere {  m*<-3.431116907908455,2.3271498147303293,-1.9692875002718282>, 1}
    sphere { m*<-2.958907389485524,-2.797333727792129,-1.6997320828929634>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.10046969645213388,0.22234276008758008,8.869444322540158>, <-1.5653872507790982,-0.23142109586474635,-1.0123510733590257>, 0.5 }
    cylinder { m*<7.254881741547835,0.13342248409322244,-5.710048967505205>, <-1.5653872507790982,-0.23142109586474635,-1.0123510733590257>, 0.5}
    cylinder { m*<-3.431116907908455,2.3271498147303293,-1.9692875002718282>, <-1.5653872507790982,-0.23142109586474635,-1.0123510733590257>, 0.5 }
    cylinder {  m*<-2.958907389485524,-2.797333727792129,-1.6997320828929634>, <-1.5653872507790982,-0.23142109586474635,-1.0123510733590257>, 0.5}

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
    sphere { m*<-1.5653872507790982,-0.23142109586474635,-1.0123510733590257>, 1 }        
    sphere {  m*<-0.10046969645213388,0.22234276008758008,8.869444322540158>, 1 }
    sphere {  m*<7.254881741547835,0.13342248409322244,-5.710048967505205>, 1 }
    sphere {  m*<-3.431116907908455,2.3271498147303293,-1.9692875002718282>, 1}
    sphere { m*<-2.958907389485524,-2.797333727792129,-1.6997320828929634>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.10046969645213388,0.22234276008758008,8.869444322540158>, <-1.5653872507790982,-0.23142109586474635,-1.0123510733590257>, 0.5 }
    cylinder { m*<7.254881741547835,0.13342248409322244,-5.710048967505205>, <-1.5653872507790982,-0.23142109586474635,-1.0123510733590257>, 0.5}
    cylinder { m*<-3.431116907908455,2.3271498147303293,-1.9692875002718282>, <-1.5653872507790982,-0.23142109586474635,-1.0123510733590257>, 0.5 }
    cylinder {  m*<-2.958907389485524,-2.797333727792129,-1.6997320828929634>, <-1.5653872507790982,-0.23142109586474635,-1.0123510733590257>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
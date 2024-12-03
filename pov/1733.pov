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
    sphere { m*<0.9767977781559772,-1.5657092046260787e-18,0.7909916313491917>, 1 }        
    sphere {  m*<1.1440519264872178,8.734160317995857e-19,3.786332160693391>, 1 }
    sphere {  m*<5.463167240149235,5.4008251271594806e-18,-1.0829763520474331>, 1 }
    sphere {  m*<-3.915763720572485,8.164965809277259,-2.27434997355192>, 1}
    sphere { m*<-3.915763720572485,-8.164965809277259,-2.2743499735519235>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1440519264872178,8.734160317995857e-19,3.786332160693391>, <0.9767977781559772,-1.5657092046260787e-18,0.7909916313491917>, 0.5 }
    cylinder { m*<5.463167240149235,5.4008251271594806e-18,-1.0829763520474331>, <0.9767977781559772,-1.5657092046260787e-18,0.7909916313491917>, 0.5}
    cylinder { m*<-3.915763720572485,8.164965809277259,-2.27434997355192>, <0.9767977781559772,-1.5657092046260787e-18,0.7909916313491917>, 0.5 }
    cylinder {  m*<-3.915763720572485,-8.164965809277259,-2.2743499735519235>, <0.9767977781559772,-1.5657092046260787e-18,0.7909916313491917>, 0.5}

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
    sphere { m*<0.9767977781559772,-1.5657092046260787e-18,0.7909916313491917>, 1 }        
    sphere {  m*<1.1440519264872178,8.734160317995857e-19,3.786332160693391>, 1 }
    sphere {  m*<5.463167240149235,5.4008251271594806e-18,-1.0829763520474331>, 1 }
    sphere {  m*<-3.915763720572485,8.164965809277259,-2.27434997355192>, 1}
    sphere { m*<-3.915763720572485,-8.164965809277259,-2.2743499735519235>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1440519264872178,8.734160317995857e-19,3.786332160693391>, <0.9767977781559772,-1.5657092046260787e-18,0.7909916313491917>, 0.5 }
    cylinder { m*<5.463167240149235,5.4008251271594806e-18,-1.0829763520474331>, <0.9767977781559772,-1.5657092046260787e-18,0.7909916313491917>, 0.5}
    cylinder { m*<-3.915763720572485,8.164965809277259,-2.27434997355192>, <0.9767977781559772,-1.5657092046260787e-18,0.7909916313491917>, 0.5 }
    cylinder {  m*<-3.915763720572485,-8.164965809277259,-2.2743499735519235>, <0.9767977781559772,-1.5657092046260787e-18,0.7909916313491917>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
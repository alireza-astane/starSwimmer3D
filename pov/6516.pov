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
    sphere { m*<-1.226413437771311,-0.7493165220158529,-0.8381665905448001>, 1 }        
    sphere {  m*<0.21837135424755694,-0.04572655236708195,9.031944714769924>, 1 }
    sphere {  m*<7.573722792247529,-0.13464682836143888,-5.547548575275426>, 1 }
    sphere {  m*<-5.1548067773774795,4.18333912230495,-2.8502695511386884>, 1}
    sphere { m*<-2.51923241393925,-3.380329132884134,-1.474343854016436>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.21837135424755694,-0.04572655236708195,9.031944714769924>, <-1.226413437771311,-0.7493165220158529,-0.8381665905448001>, 0.5 }
    cylinder { m*<7.573722792247529,-0.13464682836143888,-5.547548575275426>, <-1.226413437771311,-0.7493165220158529,-0.8381665905448001>, 0.5}
    cylinder { m*<-5.1548067773774795,4.18333912230495,-2.8502695511386884>, <-1.226413437771311,-0.7493165220158529,-0.8381665905448001>, 0.5 }
    cylinder {  m*<-2.51923241393925,-3.380329132884134,-1.474343854016436>, <-1.226413437771311,-0.7493165220158529,-0.8381665905448001>, 0.5}

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
    sphere { m*<-1.226413437771311,-0.7493165220158529,-0.8381665905448001>, 1 }        
    sphere {  m*<0.21837135424755694,-0.04572655236708195,9.031944714769924>, 1 }
    sphere {  m*<7.573722792247529,-0.13464682836143888,-5.547548575275426>, 1 }
    sphere {  m*<-5.1548067773774795,4.18333912230495,-2.8502695511386884>, 1}
    sphere { m*<-2.51923241393925,-3.380329132884134,-1.474343854016436>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.21837135424755694,-0.04572655236708195,9.031944714769924>, <-1.226413437771311,-0.7493165220158529,-0.8381665905448001>, 0.5 }
    cylinder { m*<7.573722792247529,-0.13464682836143888,-5.547548575275426>, <-1.226413437771311,-0.7493165220158529,-0.8381665905448001>, 0.5}
    cylinder { m*<-5.1548067773774795,4.18333912230495,-2.8502695511386884>, <-1.226413437771311,-0.7493165220158529,-0.8381665905448001>, 0.5 }
    cylinder {  m*<-2.51923241393925,-3.380329132884134,-1.474343854016436>, <-1.226413437771311,-0.7493165220158529,-0.8381665905448001>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
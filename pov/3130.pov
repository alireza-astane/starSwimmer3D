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
    sphere { m*<0.4132597239267699,0.9890866614949295,0.11138961899570891>, 1 }        
    sphere {  m*<0.6539948286684616,1.117796739675255,3.0989443901162597>, 1 }
    sphere {  m*<3.147968117933026,1.0911206368813042,-1.117819906455475>, 1 }
    sphere {  m*<-1.2083556359661198,3.3175606059135303,-0.8625561464202612>, 1}
    sphere { m*<-3.685680604480062,-6.759372363738557,-2.2635105407181504>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6539948286684616,1.117796739675255,3.0989443901162597>, <0.4132597239267699,0.9890866614949295,0.11138961899570891>, 0.5 }
    cylinder { m*<3.147968117933026,1.0911206368813042,-1.117819906455475>, <0.4132597239267699,0.9890866614949295,0.11138961899570891>, 0.5}
    cylinder { m*<-1.2083556359661198,3.3175606059135303,-0.8625561464202612>, <0.4132597239267699,0.9890866614949295,0.11138961899570891>, 0.5 }
    cylinder {  m*<-3.685680604480062,-6.759372363738557,-2.2635105407181504>, <0.4132597239267699,0.9890866614949295,0.11138961899570891>, 0.5}

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
    sphere { m*<0.4132597239267699,0.9890866614949295,0.11138961899570891>, 1 }        
    sphere {  m*<0.6539948286684616,1.117796739675255,3.0989443901162597>, 1 }
    sphere {  m*<3.147968117933026,1.0911206368813042,-1.117819906455475>, 1 }
    sphere {  m*<-1.2083556359661198,3.3175606059135303,-0.8625561464202612>, 1}
    sphere { m*<-3.685680604480062,-6.759372363738557,-2.2635105407181504>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6539948286684616,1.117796739675255,3.0989443901162597>, <0.4132597239267699,0.9890866614949295,0.11138961899570891>, 0.5 }
    cylinder { m*<3.147968117933026,1.0911206368813042,-1.117819906455475>, <0.4132597239267699,0.9890866614949295,0.11138961899570891>, 0.5}
    cylinder { m*<-1.2083556359661198,3.3175606059135303,-0.8625561464202612>, <0.4132597239267699,0.9890866614949295,0.11138961899570891>, 0.5 }
    cylinder {  m*<-3.685680604480062,-6.759372363738557,-2.2635105407181504>, <0.4132597239267699,0.9890866614949295,0.11138961899570891>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
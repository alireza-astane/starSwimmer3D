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
    sphere { m*<1.0578684273549983,0.37827310194609615,0.49134948655659305>, 1 }        
    sphere {  m*<1.3018938213256124,0.40817912638469656,3.4812574523930984>, 1 }
    sphere {  m*<3.795141010388147,0.40817912638469644,-0.7360247560975204>, 1 }
    sphere {  m*<-3.013509250597592,6.827842696411753,-1.9159125777486599>, 1}
    sphere { m*<-3.7893913537667943,-7.885157284010456,-2.3739887976354614>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3018938213256124,0.40817912638469656,3.4812574523930984>, <1.0578684273549983,0.37827310194609615,0.49134948655659305>, 0.5 }
    cylinder { m*<3.795141010388147,0.40817912638469644,-0.7360247560975204>, <1.0578684273549983,0.37827310194609615,0.49134948655659305>, 0.5}
    cylinder { m*<-3.013509250597592,6.827842696411753,-1.9159125777486599>, <1.0578684273549983,0.37827310194609615,0.49134948655659305>, 0.5 }
    cylinder {  m*<-3.7893913537667943,-7.885157284010456,-2.3739887976354614>, <1.0578684273549983,0.37827310194609615,0.49134948655659305>, 0.5}

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
    sphere { m*<1.0578684273549983,0.37827310194609615,0.49134948655659305>, 1 }        
    sphere {  m*<1.3018938213256124,0.40817912638469656,3.4812574523930984>, 1 }
    sphere {  m*<3.795141010388147,0.40817912638469644,-0.7360247560975204>, 1 }
    sphere {  m*<-3.013509250597592,6.827842696411753,-1.9159125777486599>, 1}
    sphere { m*<-3.7893913537667943,-7.885157284010456,-2.3739887976354614>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3018938213256124,0.40817912638469656,3.4812574523930984>, <1.0578684273549983,0.37827310194609615,0.49134948655659305>, 0.5 }
    cylinder { m*<3.795141010388147,0.40817912638469644,-0.7360247560975204>, <1.0578684273549983,0.37827310194609615,0.49134948655659305>, 0.5}
    cylinder { m*<-3.013509250597592,6.827842696411753,-1.9159125777486599>, <1.0578684273549983,0.37827310194609615,0.49134948655659305>, 0.5 }
    cylinder {  m*<-3.7893913537667943,-7.885157284010456,-2.3739887976354614>, <1.0578684273549983,0.37827310194609615,0.49134948655659305>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
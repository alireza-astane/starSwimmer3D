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
    sphere { m*<-0.178932587103399,-0.09088489053803529,-0.576278164995421>, 1 }        
    sphere {  m*<0.2269467994910928,0.12612030127399296,4.460739179113826>, 1 }
    sphere {  m*<2.555775806902858,0.011149084848338875,-1.805487690446603>, 1 }
    sphere {  m*<-1.800547946996289,2.237589053880564,-1.5502239304113898>, 1}
    sphere { m*<-1.5327607259584572,-2.6501028885233335,-1.3606776452488172>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2269467994910928,0.12612030127399296,4.460739179113826>, <-0.178932587103399,-0.09088489053803529,-0.576278164995421>, 0.5 }
    cylinder { m*<2.555775806902858,0.011149084848338875,-1.805487690446603>, <-0.178932587103399,-0.09088489053803529,-0.576278164995421>, 0.5}
    cylinder { m*<-1.800547946996289,2.237589053880564,-1.5502239304113898>, <-0.178932587103399,-0.09088489053803529,-0.576278164995421>, 0.5 }
    cylinder {  m*<-1.5327607259584572,-2.6501028885233335,-1.3606776452488172>, <-0.178932587103399,-0.09088489053803529,-0.576278164995421>, 0.5}

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
    sphere { m*<-0.178932587103399,-0.09088489053803529,-0.576278164995421>, 1 }        
    sphere {  m*<0.2269467994910928,0.12612030127399296,4.460739179113826>, 1 }
    sphere {  m*<2.555775806902858,0.011149084848338875,-1.805487690446603>, 1 }
    sphere {  m*<-1.800547946996289,2.237589053880564,-1.5502239304113898>, 1}
    sphere { m*<-1.5327607259584572,-2.6501028885233335,-1.3606776452488172>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2269467994910928,0.12612030127399296,4.460739179113826>, <-0.178932587103399,-0.09088489053803529,-0.576278164995421>, 0.5 }
    cylinder { m*<2.555775806902858,0.011149084848338875,-1.805487690446603>, <-0.178932587103399,-0.09088489053803529,-0.576278164995421>, 0.5}
    cylinder { m*<-1.800547946996289,2.237589053880564,-1.5502239304113898>, <-0.178932587103399,-0.09088489053803529,-0.576278164995421>, 0.5 }
    cylinder {  m*<-1.5327607259584572,-2.6501028885233335,-1.3606776452488172>, <-0.178932587103399,-0.09088489053803529,-0.576278164995421>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
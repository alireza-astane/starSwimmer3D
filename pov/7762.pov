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
    sphere { m*<-0.4394321647967976,-0.4775376846782703,-0.45305957018869303>, 1 }        
    sphere {  m*<0.979735329403364,0.5124012292016469,9.396230526846454>, 1 }
    sphere {  m*<8.347522527726161,0.22730897840938447,-5.174446902227477>, 1 }
    sphere {  m*<-6.548440665962832,6.7503903520300215,-3.6836399990458712>, 1}
    sphere { m*<-3.856834689919979,-7.91997310984533,-2.0356161307623957>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.979735329403364,0.5124012292016469,9.396230526846454>, <-0.4394321647967976,-0.4775376846782703,-0.45305957018869303>, 0.5 }
    cylinder { m*<8.347522527726161,0.22730897840938447,-5.174446902227477>, <-0.4394321647967976,-0.4775376846782703,-0.45305957018869303>, 0.5}
    cylinder { m*<-6.548440665962832,6.7503903520300215,-3.6836399990458712>, <-0.4394321647967976,-0.4775376846782703,-0.45305957018869303>, 0.5 }
    cylinder {  m*<-3.856834689919979,-7.91997310984533,-2.0356161307623957>, <-0.4394321647967976,-0.4775376846782703,-0.45305957018869303>, 0.5}

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
    sphere { m*<-0.4394321647967976,-0.4775376846782703,-0.45305957018869303>, 1 }        
    sphere {  m*<0.979735329403364,0.5124012292016469,9.396230526846454>, 1 }
    sphere {  m*<8.347522527726161,0.22730897840938447,-5.174446902227477>, 1 }
    sphere {  m*<-6.548440665962832,6.7503903520300215,-3.6836399990458712>, 1}
    sphere { m*<-3.856834689919979,-7.91997310984533,-2.0356161307623957>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.979735329403364,0.5124012292016469,9.396230526846454>, <-0.4394321647967976,-0.4775376846782703,-0.45305957018869303>, 0.5 }
    cylinder { m*<8.347522527726161,0.22730897840938447,-5.174446902227477>, <-0.4394321647967976,-0.4775376846782703,-0.45305957018869303>, 0.5}
    cylinder { m*<-6.548440665962832,6.7503903520300215,-3.6836399990458712>, <-0.4394321647967976,-0.4775376846782703,-0.45305957018869303>, 0.5 }
    cylinder {  m*<-3.856834689919979,-7.91997310984533,-2.0356161307623957>, <-0.4394321647967976,-0.4775376846782703,-0.45305957018869303>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
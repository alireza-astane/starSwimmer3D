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
    sphere { m*<0.3503725038725784,0.8702073881315773,0.07495316054921336>, 1 }        
    sphere {  m*<0.5911076086142703,0.9989174663119029,3.062507931669766>, 1 }
    sphere {  m*<3.0850808978788358,0.9722413635179519,-1.1542563649019701>, 1 }
    sphere {  m*<-1.2712428560203115,3.198681332550181,-0.8989926048667561>, 1}
    sphere { m*<-3.4800585769946673,-6.370673400834822,-2.1443744355810566>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5911076086142703,0.9989174663119029,3.062507931669766>, <0.3503725038725784,0.8702073881315773,0.07495316054921336>, 0.5 }
    cylinder { m*<3.0850808978788358,0.9722413635179519,-1.1542563649019701>, <0.3503725038725784,0.8702073881315773,0.07495316054921336>, 0.5}
    cylinder { m*<-1.2712428560203115,3.198681332550181,-0.8989926048667561>, <0.3503725038725784,0.8702073881315773,0.07495316054921336>, 0.5 }
    cylinder {  m*<-3.4800585769946673,-6.370673400834822,-2.1443744355810566>, <0.3503725038725784,0.8702073881315773,0.07495316054921336>, 0.5}

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
    sphere { m*<0.3503725038725784,0.8702073881315773,0.07495316054921336>, 1 }        
    sphere {  m*<0.5911076086142703,0.9989174663119029,3.062507931669766>, 1 }
    sphere {  m*<3.0850808978788358,0.9722413635179519,-1.1542563649019701>, 1 }
    sphere {  m*<-1.2712428560203115,3.198681332550181,-0.8989926048667561>, 1}
    sphere { m*<-3.4800585769946673,-6.370673400834822,-2.1443744355810566>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5911076086142703,0.9989174663119029,3.062507931669766>, <0.3503725038725784,0.8702073881315773,0.07495316054921336>, 0.5 }
    cylinder { m*<3.0850808978788358,0.9722413635179519,-1.1542563649019701>, <0.3503725038725784,0.8702073881315773,0.07495316054921336>, 0.5}
    cylinder { m*<-1.2712428560203115,3.198681332550181,-0.8989926048667561>, <0.3503725038725784,0.8702073881315773,0.07495316054921336>, 0.5 }
    cylinder {  m*<-3.4800585769946673,-6.370673400834822,-2.1443744355810566>, <0.3503725038725784,0.8702073881315773,0.07495316054921336>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
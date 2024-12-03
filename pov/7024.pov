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
    sphere { m*<-0.7944265207408138,-1.2506460765019343,-0.6174530365233587>, 1 }        
    sphere {  m*<0.6247409734593493,-0.2607071626220161,9.231837060511799>, 1 }
    sphere {  m*<7.992528171782158,-0.545799413414278,-5.338840368562143>, 1 }
    sphere {  m*<-6.903435021906845,5.977281960206378,-3.848033465380537>, 1}
    sphere { m*<-2.095260115905195,-4.083607648844028,-1.219852878247553>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6247409734593493,-0.2607071626220161,9.231837060511799>, <-0.7944265207408138,-1.2506460765019343,-0.6174530365233587>, 0.5 }
    cylinder { m*<7.992528171782158,-0.545799413414278,-5.338840368562143>, <-0.7944265207408138,-1.2506460765019343,-0.6174530365233587>, 0.5}
    cylinder { m*<-6.903435021906845,5.977281960206378,-3.848033465380537>, <-0.7944265207408138,-1.2506460765019343,-0.6174530365233587>, 0.5 }
    cylinder {  m*<-2.095260115905195,-4.083607648844028,-1.219852878247553>, <-0.7944265207408138,-1.2506460765019343,-0.6174530365233587>, 0.5}

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
    sphere { m*<-0.7944265207408138,-1.2506460765019343,-0.6174530365233587>, 1 }        
    sphere {  m*<0.6247409734593493,-0.2607071626220161,9.231837060511799>, 1 }
    sphere {  m*<7.992528171782158,-0.545799413414278,-5.338840368562143>, 1 }
    sphere {  m*<-6.903435021906845,5.977281960206378,-3.848033465380537>, 1}
    sphere { m*<-2.095260115905195,-4.083607648844028,-1.219852878247553>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6247409734593493,-0.2607071626220161,9.231837060511799>, <-0.7944265207408138,-1.2506460765019343,-0.6174530365233587>, 0.5 }
    cylinder { m*<7.992528171782158,-0.545799413414278,-5.338840368562143>, <-0.7944265207408138,-1.2506460765019343,-0.6174530365233587>, 0.5}
    cylinder { m*<-6.903435021906845,5.977281960206378,-3.848033465380537>, <-0.7944265207408138,-1.2506460765019343,-0.6174530365233587>, 0.5 }
    cylinder {  m*<-2.095260115905195,-4.083607648844028,-1.219852878247553>, <-0.7944265207408138,-1.2506460765019343,-0.6174530365233587>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
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
    sphere { m*<-0.7263191298563287,-1.102321477655879,-0.5859133517846069>, 1 }        
    sphere {  m*<0.692848364343834,-0.11238256377596123,9.263376745250547>, 1 }
    sphere {  m*<8.060635562666631,-0.397474814568224,-5.307300683823388>, 1 }
    sphere {  m*<-6.835327631022357,6.1256065590524305,-3.8164937806417836>, 1}
    sphere { m*<-2.4688215803237656,-4.897151577552235,-1.3928445413446071>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.692848364343834,-0.11238256377596123,9.263376745250547>, <-0.7263191298563287,-1.102321477655879,-0.5859133517846069>, 0.5 }
    cylinder { m*<8.060635562666631,-0.397474814568224,-5.307300683823388>, <-0.7263191298563287,-1.102321477655879,-0.5859133517846069>, 0.5}
    cylinder { m*<-6.835327631022357,6.1256065590524305,-3.8164937806417836>, <-0.7263191298563287,-1.102321477655879,-0.5859133517846069>, 0.5 }
    cylinder {  m*<-2.4688215803237656,-4.897151577552235,-1.3928445413446071>, <-0.7263191298563287,-1.102321477655879,-0.5859133517846069>, 0.5}

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
    sphere { m*<-0.7263191298563287,-1.102321477655879,-0.5859133517846069>, 1 }        
    sphere {  m*<0.692848364343834,-0.11238256377596123,9.263376745250547>, 1 }
    sphere {  m*<8.060635562666631,-0.397474814568224,-5.307300683823388>, 1 }
    sphere {  m*<-6.835327631022357,6.1256065590524305,-3.8164937806417836>, 1}
    sphere { m*<-2.4688215803237656,-4.897151577552235,-1.3928445413446071>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.692848364343834,-0.11238256377596123,9.263376745250547>, <-0.7263191298563287,-1.102321477655879,-0.5859133517846069>, 0.5 }
    cylinder { m*<8.060635562666631,-0.397474814568224,-5.307300683823388>, <-0.7263191298563287,-1.102321477655879,-0.5859133517846069>, 0.5}
    cylinder { m*<-6.835327631022357,6.1256065590524305,-3.8164937806417836>, <-0.7263191298563287,-1.102321477655879,-0.5859133517846069>, 0.5 }
    cylinder {  m*<-2.4688215803237656,-4.897151577552235,-1.3928445413446071>, <-0.7263191298563287,-1.102321477655879,-0.5859133517846069>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
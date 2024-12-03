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
    sphere { m*<-1.2595595794512775,-0.17454569221632774,-1.2112251001322833>, 1 }        
    sphere {  m*<0.06648173640848282,0.28090823130217657,8.689961377230146>, 1 }
    sphere {  m*<6.154387108904549,0.08348427617938875,-5.024350146371441>, 1 }
    sphere {  m*<-2.925596103636165,2.1545857364321748,-2.105349542676216>, 1}
    sphere { m*<-2.6578088825983337,-2.7331062059717226,-1.9158032575136452>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.06648173640848282,0.28090823130217657,8.689961377230146>, <-1.2595595794512775,-0.17454569221632774,-1.2112251001322833>, 0.5 }
    cylinder { m*<6.154387108904549,0.08348427617938875,-5.024350146371441>, <-1.2595595794512775,-0.17454569221632774,-1.2112251001322833>, 0.5}
    cylinder { m*<-2.925596103636165,2.1545857364321748,-2.105349542676216>, <-1.2595595794512775,-0.17454569221632774,-1.2112251001322833>, 0.5 }
    cylinder {  m*<-2.6578088825983337,-2.7331062059717226,-1.9158032575136452>, <-1.2595595794512775,-0.17454569221632774,-1.2112251001322833>, 0.5}

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
    sphere { m*<-1.2595595794512775,-0.17454569221632774,-1.2112251001322833>, 1 }        
    sphere {  m*<0.06648173640848282,0.28090823130217657,8.689961377230146>, 1 }
    sphere {  m*<6.154387108904549,0.08348427617938875,-5.024350146371441>, 1 }
    sphere {  m*<-2.925596103636165,2.1545857364321748,-2.105349542676216>, 1}
    sphere { m*<-2.6578088825983337,-2.7331062059717226,-1.9158032575136452>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.06648173640848282,0.28090823130217657,8.689961377230146>, <-1.2595595794512775,-0.17454569221632774,-1.2112251001322833>, 0.5 }
    cylinder { m*<6.154387108904549,0.08348427617938875,-5.024350146371441>, <-1.2595595794512775,-0.17454569221632774,-1.2112251001322833>, 0.5}
    cylinder { m*<-2.925596103636165,2.1545857364321748,-2.105349542676216>, <-1.2595595794512775,-0.17454569221632774,-1.2112251001322833>, 0.5 }
    cylinder {  m*<-2.6578088825983337,-2.7331062059717226,-1.9158032575136452>, <-1.2595595794512775,-0.17454569221632774,-1.2112251001322833>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
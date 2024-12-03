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
    sphere { m*<-0.774601688978481,-1.2074714638769133,-0.6082724037658469>, 1 }        
    sphere {  m*<0.6445658052216813,-0.21753254999699556,9.241017693269304>, 1 }
    sphere {  m*<8.012353003544488,-0.5026248007892573,-5.32965973580463>, 1 }
    sphere {  m*<-6.883610190144512,6.020456572831399,-3.838852832623024>, 1}
    sphere { m*<-2.207362344817801,-4.327744414579372,-1.2717660247137235>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6445658052216813,-0.21753254999699556,9.241017693269304>, <-0.774601688978481,-1.2074714638769133,-0.6082724037658469>, 0.5 }
    cylinder { m*<8.012353003544488,-0.5026248007892573,-5.32965973580463>, <-0.774601688978481,-1.2074714638769133,-0.6082724037658469>, 0.5}
    cylinder { m*<-6.883610190144512,6.020456572831399,-3.838852832623024>, <-0.774601688978481,-1.2074714638769133,-0.6082724037658469>, 0.5 }
    cylinder {  m*<-2.207362344817801,-4.327744414579372,-1.2717660247137235>, <-0.774601688978481,-1.2074714638769133,-0.6082724037658469>, 0.5}

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
    sphere { m*<-0.774601688978481,-1.2074714638769133,-0.6082724037658469>, 1 }        
    sphere {  m*<0.6445658052216813,-0.21753254999699556,9.241017693269304>, 1 }
    sphere {  m*<8.012353003544488,-0.5026248007892573,-5.32965973580463>, 1 }
    sphere {  m*<-6.883610190144512,6.020456572831399,-3.838852832623024>, 1}
    sphere { m*<-2.207362344817801,-4.327744414579372,-1.2717660247137235>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6445658052216813,-0.21753254999699556,9.241017693269304>, <-0.774601688978481,-1.2074714638769133,-0.6082724037658469>, 0.5 }
    cylinder { m*<8.012353003544488,-0.5026248007892573,-5.32965973580463>, <-0.774601688978481,-1.2074714638769133,-0.6082724037658469>, 0.5}
    cylinder { m*<-6.883610190144512,6.020456572831399,-3.838852832623024>, <-0.774601688978481,-1.2074714638769133,-0.6082724037658469>, 0.5 }
    cylinder {  m*<-2.207362344817801,-4.327744414579372,-1.2717660247137235>, <-0.774601688978481,-1.2074714638769133,-0.6082724037658469>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
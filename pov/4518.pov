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
    sphere { m*<-0.20518056184182087,-0.10491848519919814,-0.9020190337655392>, 1 }        
    sphere {  m*<0.327084754742059,0.17965949765750083,5.703464565181997>, 1 }
    sphere {  m*<2.5295278321644363,-0.0028845098128239266,-2.13122855921672>, 1 }
    sphere {  m*<-1.8267959217347107,2.2235554592194013,-1.875964799181507>, 1}
    sphere { m*<-1.559008700696879,-2.664136483184496,-1.686418514018934>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.327084754742059,0.17965949765750083,5.703464565181997>, <-0.20518056184182087,-0.10491848519919814,-0.9020190337655392>, 0.5 }
    cylinder { m*<2.5295278321644363,-0.0028845098128239266,-2.13122855921672>, <-0.20518056184182087,-0.10491848519919814,-0.9020190337655392>, 0.5}
    cylinder { m*<-1.8267959217347107,2.2235554592194013,-1.875964799181507>, <-0.20518056184182087,-0.10491848519919814,-0.9020190337655392>, 0.5 }
    cylinder {  m*<-1.559008700696879,-2.664136483184496,-1.686418514018934>, <-0.20518056184182087,-0.10491848519919814,-0.9020190337655392>, 0.5}

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
    sphere { m*<-0.20518056184182087,-0.10491848519919814,-0.9020190337655392>, 1 }        
    sphere {  m*<0.327084754742059,0.17965949765750083,5.703464565181997>, 1 }
    sphere {  m*<2.5295278321644363,-0.0028845098128239266,-2.13122855921672>, 1 }
    sphere {  m*<-1.8267959217347107,2.2235554592194013,-1.875964799181507>, 1}
    sphere { m*<-1.559008700696879,-2.664136483184496,-1.686418514018934>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.327084754742059,0.17965949765750083,5.703464565181997>, <-0.20518056184182087,-0.10491848519919814,-0.9020190337655392>, 0.5 }
    cylinder { m*<2.5295278321644363,-0.0028845098128239266,-2.13122855921672>, <-0.20518056184182087,-0.10491848519919814,-0.9020190337655392>, 0.5}
    cylinder { m*<-1.8267959217347107,2.2235554592194013,-1.875964799181507>, <-0.20518056184182087,-0.10491848519919814,-0.9020190337655392>, 0.5 }
    cylinder {  m*<-1.559008700696879,-2.664136483184496,-1.686418514018934>, <-0.20518056184182087,-0.10491848519919814,-0.9020190337655392>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
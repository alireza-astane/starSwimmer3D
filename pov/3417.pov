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
    sphere { m*<0.2054315073412007,0.5962172143989433,-0.00902473818376126>, 1 }        
    sphere {  m*<0.4461666120828923,0.7249272925792688,2.9785300329367894>, 1 }
    sphere {  m*<2.9401399013474587,0.6982511897853176,-1.2382342636349457>, 1 }
    sphere {  m*<-1.4161838525516899,2.924691158817545,-0.9829705035997315>, 1}
    sphere { m*<-2.9868952440790237,-5.438418824850379,-1.8586387126212625>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4461666120828923,0.7249272925792688,2.9785300329367894>, <0.2054315073412007,0.5962172143989433,-0.00902473818376126>, 0.5 }
    cylinder { m*<2.9401399013474587,0.6982511897853176,-1.2382342636349457>, <0.2054315073412007,0.5962172143989433,-0.00902473818376126>, 0.5}
    cylinder { m*<-1.4161838525516899,2.924691158817545,-0.9829705035997315>, <0.2054315073412007,0.5962172143989433,-0.00902473818376126>, 0.5 }
    cylinder {  m*<-2.9868952440790237,-5.438418824850379,-1.8586387126212625>, <0.2054315073412007,0.5962172143989433,-0.00902473818376126>, 0.5}

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
    sphere { m*<0.2054315073412007,0.5962172143989433,-0.00902473818376126>, 1 }        
    sphere {  m*<0.4461666120828923,0.7249272925792688,2.9785300329367894>, 1 }
    sphere {  m*<2.9401399013474587,0.6982511897853176,-1.2382342636349457>, 1 }
    sphere {  m*<-1.4161838525516899,2.924691158817545,-0.9829705035997315>, 1}
    sphere { m*<-2.9868952440790237,-5.438418824850379,-1.8586387126212625>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4461666120828923,0.7249272925792688,2.9785300329367894>, <0.2054315073412007,0.5962172143989433,-0.00902473818376126>, 0.5 }
    cylinder { m*<2.9401399013474587,0.6982511897853176,-1.2382342636349457>, <0.2054315073412007,0.5962172143989433,-0.00902473818376126>, 0.5}
    cylinder { m*<-1.4161838525516899,2.924691158817545,-0.9829705035997315>, <0.2054315073412007,0.5962172143989433,-0.00902473818376126>, 0.5 }
    cylinder {  m*<-2.9868952440790237,-5.438418824850379,-1.8586387126212625>, <0.2054315073412007,0.5962172143989433,-0.00902473818376126>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    